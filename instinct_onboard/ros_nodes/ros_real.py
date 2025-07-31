import re

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from unitree_go.msg import WirelessController
from unitree_hg.msg import IMUState, LowCmd, LowState  # MotorState,; MotorCmd,

import instinct_onboard.robot_cfgs as robot_cfgs
from crc_module import get_crc
from instinct_onboard import utils


class Ros2Real(Node):
    """This is the basic implementation of handling ROS messages matching the design of IsaacLab.
    It is designed to be used in the script directly to run the ONNX function. But please handle the
    impl of combining observations in the agent implementation.
    """

    def __init__(
        self,
        low_state_topic: str = "/lowstate",
        low_cmd_topic: str = "/lowcmd",
        imu_state_topic: str = "/secondary_imu",
        joy_stick_topic: str = "/wirelesscontroller",
        cfg: dict = dict(),  # default env cfg to read from. Typically from a env.yaml in your logdir
        computer_clip_torque=True,  # if True, the actions will be clipped by torque limits
        joint_pos_protect_ratio=1.5,  # if the joint_pos is out of the range of this ratio, the process will shutdown.
        kp_factor=1.0,  # the factor to multiply the p_gain
        kd_factor=1.0,  # the factor to multiply the d_gain
        torque_limits_ratio=1.0,  # the factor to multiply the torque limits
        robot_class_name="G1_29Dof",
        dryrun=True,  # if True, the robot will not send commands to the real robot
    ):
        super().__init__("unitree_ros_real")
        self.NUM_JOINTS = getattr(robot_cfgs, robot_class_name).NUM_JOINTS
        self.NUM_ACTIONS = getattr(robot_cfgs, robot_class_name).NUM_ACTIONS
        self.low_state_topic = low_state_topic
        self.imu_state_topic = imu_state_topic
        # Generate a unique cmd topic so that the low_cmd will not send to the robot's motor.
        self.low_cmd_topic = (
            low_cmd_topic if not dryrun else low_cmd_topic + "_dryrun_" + str(np.random.randint(0, 65535))
        )
        self.joy_stick_topic = joy_stick_topic
        self.cfg = cfg
        self.computer_clip_torque = computer_clip_torque
        self.joint_pos_protect_ratio = joint_pos_protect_ratio
        self.kp_factor = kp_factor
        self.kd_factor = kd_factor
        self.torque_limits_ratio = torque_limits_ratio
        self.robot_class_name = robot_class_name
        self.dryrun = dryrun

        # load robot-specific configurations
        self.joint_map = getattr(robot_cfgs, robot_class_name).joint_map
        self.sim_joint_names = getattr(robot_cfgs, robot_class_name).sim_joint_names
        self.real_joint_names = getattr(robot_cfgs, robot_class_name).real_joint_names
        self.joint_signs = getattr(robot_cfgs, robot_class_name).joint_signs
        self.turn_on_motor_mode = getattr(robot_cfgs, robot_class_name).turn_on_motor_mode
        self.mode_pr = getattr(robot_cfgs, robot_class_name).mode_pr

        self.parse_config()

    def parse_config(self):
        """Parse, set attributes from config dict, initialize buffers to speed up the computation"""
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.gravity_vec = np.zeros(3)
        self.gravity_vec[self.up_axis_idx] = -1

        # controls
        self.default_joint_pos = np.zeros(self.NUM_JOINTS, dtype=np.float32)
        for joint_name_expr, joint_pos in self.cfg["scene"]["robot"]["init_state"]["joint_pos"].items():
            # compute the default joint pos from configuration for articulation.default_joint_pos
            for i in range(self.NUM_JOINTS):
                name = self.sim_joint_names[i]
                if re.search(joint_name_expr, name):
                    self.default_joint_pos[i] = joint_pos
        self.p_gains = np.zeros(self.NUM_JOINTS, dtype=np.float32)
        self.d_gains = np.zeros(self.NUM_JOINTS, dtype=np.float32)
        for actuator_name, actuator_config in self.cfg["scene"]["robot"]["actuators"].items():
            assert (
                "PDActuator" in actuator_config["class_type"]
            ), "Only PDActuator trained model is supported for now. Get {} in actuator {}".format(
                actuator_config["class_type"], actuator_name
            )
            for i in range(self.NUM_JOINTS):
                name = self.sim_joint_names[i]
                for _, joint_name_expr in enumerate(actuator_config["joint_names_expr"]):
                    if re.search(joint_name_expr, name):
                        if isinstance(actuator_config["stiffness"], dict):
                            for key, value in actuator_config["stiffness"].items():
                                if re.search(key, name):
                                    self.p_gains[i] = value
                        else:
                            self.p_gains[i] = actuator_config["stiffness"]
                        if isinstance(actuator_config["damping"], dict):
                            for key, value in actuator_config["damping"].items():
                                if re.search(key, name):
                                    self.d_gains[i] = value
                        else:
                            self.d_gains[i] = actuator_config["damping"]
                        # print(f"Joint {i}({self.sim_joint_names[i]}) has p_gain {self.p_gains[i]} and d_gain {self.d_gains[i]}")
        self.p_gains *= self.kp_factor
        self.d_gains *= self.kd_factor
        self.get_logger().info(f"default_joint_pos: {self.default_joint_pos}")
        self.get_logger().info(f"PD gains are set to: p_gains: {self.p_gains}, d_gains: {self.d_gains}")
        self.get_logger().info(f"PD gains are set by kp_factor: {self.kp_factor}, kd_factor: {self.kd_factor}")
        self.torque_limits = getattr(robot_cfgs, self.robot_class_name).torque_limits * self.torque_limits_ratio
        self.get_logger().info(f"Torque limits are set by ratio of : {self.torque_limits_ratio}")

        # buffers for observation output (in simulation order)
        self.joint_pos_ = np.zeros(
            self.NUM_JOINTS, dtype=np.float32
        )  # in robot urdf coordinate, but in simulation order. no offset subtracted
        self.joint_vel_ = np.zeros(self.NUM_JOINTS, dtype=np.float32)

        # actions
        self.action_scale = np.zeros(self.NUM_ACTIONS, dtype=np.float32)
        for action_names, action_config in self.cfg["actions"].items():
            if not action_config["asset_name"] == "robot":
                continue
            for i in range(self.NUM_JOINTS):
                name = self.sim_joint_names[i]
                for _, joint_name_expr in enumerate(action_config["joint_names"]):
                    if re.search(joint_name_expr, name):
                        self.action_scale[i] = action_config["scale"]
                        # print("Joint {}({}) has action scale {}".format(i, name, self.action_scale[i]))
                    if not action_config["use_default_offset"]:
                        # not using articulation.default_joint_pos as default offset
                        if isinstance(action_config["offset"], dict):
                            self.default_joint_pos[i] = action_config["offset"][joint_name_expr]
                        else:
                            self.default_joint_pos[i] = action_config["offset"]
        self.actions_raw = np.zeros(self.NUM_ACTIONS, dtype=np.float32)
        self.actions = np.zeros(self.NUM_ACTIONS, dtype=np.float32)

        # hardware related, in simulation order
        self.joint_limits_high = getattr(robot_cfgs, self.robot_class_name).joint_limits_high
        self.joint_limits_low = getattr(robot_cfgs, self.robot_class_name).joint_limits_low
        joint_pos_mid = (self.joint_limits_high + self.joint_limits_low) / 2
        joint_pos_range = (self.joint_limits_high - self.joint_limits_low) / 2
        self.joint_pos_protect_high = joint_pos_mid + joint_pos_range * self.joint_pos_protect_ratio
        self.joint_pos_protect_low = joint_pos_mid - joint_pos_range * self.joint_pos_protect_ratio

    def start_ros_handlers(self):
        """After initializing the env and policy, register ros related callbacks and topics"""
        # ROS publishers
        self.action_publisher = self.create_publisher(Float32MultiArray, "/raw_actions", 1)
        self.low_cmd_publisher = self.create_publisher(LowCmd, self.low_cmd_topic, 1)
        self.low_cmd_buffer = LowCmd()
        self.low_cmd_buffer.mode_pr = self.mode_pr

        # ROS subscribers
        self.low_state_subscriber = self.create_subscription(
            LowState, self.low_state_topic, self._low_state_callback, 1
        )
        self.torso_imu_subscriber = self.create_subscription(
            IMUState, self.imu_state_topic, self._torso_imu_state_callback, 1
        )
        self.joy_stick_subscriber = self.create_subscription(
            WirelessController, self.joy_stick_topic, self._joy_stick_callback, 1
        )
        self.get_logger().info(
            "ROS handlers started, waiting to receive critical low state and wireless controller messages."
        )
        if not self.dryrun:
            self.get_logger().warn(
                f"You are running the code in no-dryrun mode and publishing to '{self.low_cmd_topic}', Please keep"
                " safe."
            )
        else:
            self.get_logger().warn(
                f"You are publishing low cmd to '{self.low_cmd_topic}' because of dryrun mode, Please check and be"
                " safe."
            )
        while rclpy.ok():
            rclpy.spin_once(self)
            if self.check_buffers_ready():
                break
        self.get_logger().info("All necessary buffers received, the robot is ready to go.")

    def check_buffers_ready(self):
        """Check if all the necessary buffers are ready to use. Only used at the the end of the start_ros_handlers."""
        buffer_ready = hasattr(self, "low_state_buffer") and hasattr(self, "joy_stick_buffer")
        if self.imu_state_topic is not None:
            buffer_ready = buffer_ready and hasattr(self, "torso_imu_buffer")
        return buffer_ready

    def get_cfg_main_loop_duration(self):
        """Get the main running frequency based on self.cfg, which is typically from env.yaml."""
        return self.cfg["sim"]["dt"] * self.cfg["decimation"]

    """
    ROS callbacks and handlers that update the buffer
    """

    def _low_state_callback(self, msg):
        """store and handle proprioception data"""
        self.get_logger().info("Low state data received.", once=True)
        self.low_state_buffer = msg  # keep the latest low state
        self.low_cmd_buffer.mode_machine = msg.mode_machine

        # refresh joint_pos and joint_vel
        for sim_idx in range(self.NUM_JOINTS):
            real_idx = self.joint_map[sim_idx]
            self.joint_pos_[sim_idx] = self.low_state_buffer.motor_state[real_idx].q * self.joint_signs[sim_idx]
        for sim_idx in range(self.NUM_JOINTS):
            real_idx = self.joint_map[sim_idx]
            self.joint_vel_[sim_idx] = self.low_state_buffer.motor_state[real_idx].dq * self.joint_signs[sim_idx]
        # automatic safety check
        for sim_idx in range(self.NUM_JOINTS):
            real_idx = self.joint_map[sim_idx]
            if (
                self.joint_pos_[sim_idx] > self.joint_pos_protect_high[sim_idx]
                or self.joint_pos_[sim_idx] < self.joint_pos_protect_low[sim_idx]
            ):
                self.get_logger().error(
                    f"Joint {sim_idx}(sim), {real_idx}(real) position out of range at"
                    f" {self.low_state_buffer.motor_state[real_idx].q}"
                )
                self.get_logger().error("The motors and this process shuts down.")
                self._turn_off_motors()
                raise SystemExit()

    def _torso_imu_state_callback(self, msg):
        """store and handle torso imu data"""
        self.get_logger().info("Torso IMU data received.", once=True)
        self.torso_imu_buffer = msg

    def _joy_stick_callback(self, msg):
        self.get_logger().info("Wireless controller data received.", once=True)
        self.joy_stick_buffer = msg

        # refer to Unitree Remote Control data structure, msg.keys is a bit mask
        # 00000000 00000001 means pressing the 0-th button (R1)
        # 00000000 00000010 means pressing the 1-th button (L1)
        # 10000000 00000000 means pressing the 15-th button (left)
        if (msg.keys & robot_cfgs.WirelessButtons.R2) or (
            msg.keys & robot_cfgs.WirelessButtons.L2
        ):  # R2 or L2 is pressed
            self.get_logger().warn("R2 or L2 is pressed, the motors and this process shuts down.")
            self._turn_off_motors()
            raise SystemExit()

    """
    Refresh observation buffer and corresponding sub-functions
    NOTE: everything will be NON-batchwise. There is NO batch dimension in the observation.
    """

    def _get_quat_w_obs(self):
        """Get the quaternion in wxyz format from the torso IMU or low state buffer."""
        if hasattr(self, "torso_imu_buffer"):
            return np.array(self.torso_imu_buffer.quaternion, dtype=np.float32)
        else:
            return np.array(self.low_state_buffer.imu_state.quaternion, dtype=np.float32)

    def _get_ang_vel_obs(self):
        if hasattr(self, "torso_imu_buffer"):
            return np.array(self.torso_imu_buffer.gyroscope, dtype=np.float32)
        else:
            return np.array(self.low_state_buffer.imu_state.gyroscope, dtype=np.float32)

    def _get_projected_gravity_obs(self):
        if hasattr(self, "torso_imu_buffer"):
            quat_wxyz = np.quaternion(
                self.torso_imu_buffer.quaternion[0],
                self.torso_imu_buffer.quaternion[1],
                self.torso_imu_buffer.quaternion[2],
                self.torso_imu_buffer.quaternion[3],
            )
        else:
            quat_wxyz = np.quaternion(
                self.low_state_buffer.imu_state.quaternion[0],
                self.low_state_buffer.imu_state.quaternion[1],
                self.low_state_buffer.imu_state.quaternion[2],
                self.low_state_buffer.imu_state.quaternion[3],
            )
        return utils.quat_rotate_inverse(
            quat_wxyz,
            self.gravity_vec,
        ).astype(np.float32)

    def _get_joint_pos_obs(self):
        return self.joint_pos_

    def _get_joint_pos_rel_obs(self):
        return self.joint_pos_ - self.default_joint_pos

    def _get_joint_vel_obs(self):
        return self.joint_vel_

    def _get_last_actions_obs(self):
        return self.actions

    """
    Control related functions
    """

    def clip_by_torque_limit(self, actions_scaled):
        """Different from simulation, we reverse the process and clip the actions directly,
        so that the PD controller runs in robot but not our script.
        """

        # NOTE: Currently only support position control with PD controller
        p_limits_low = (-self.torque_limits) + self.d_gains * self.joint_vel_
        p_limits_high = (self.torque_limits) + self.d_gains * self.joint_vel_
        actions_low = (p_limits_low / self.p_gains) - self.default_joint_pos + self.joint_pos_
        actions_high = (p_limits_high / self.p_gains) - self.default_joint_pos + self.joint_pos_

        return np.clip(actions_scaled, actions_low, actions_high)

    def send_action(self, actions):
        """Send the action to the robot motors, which does the preprocessing
        just like env.step in simulation.
        Thus, the actions has the batch dimension, whose size is 1.
        """
        # NOTE: Only calling this function currently will update self.action for self._get_last_actions_obs
        self.actions[:] = actions
        self.action_publisher.publish(Float32MultiArray(data=actions))
        if self.computer_clip_torque:
            clipped_scaled_action = self.clip_by_torque_limit(actions * self.action_scale)
            clipped_joints = np.where(clipped_scaled_action != actions * self.action_scale)[0]
            if len(clipped_joints) > 0:
                self.get_logger().warn(
                    f"Computer Clip Torque is True, the following joints (sim) are clipped: {clipped_joints}",
                    throttle_duration_sec=5,
                )
        else:
            self.get_logger().warn("Computer Clip Torque is False, the robot may be damaged.", throttle_duration_sec=5)
            clipped_scaled_action = actions * self.action_scale
        robot_coordinates_action = clipped_scaled_action + self.default_joint_pos
        if np.isnan(actions).any():
            self.get_logger().error("Actions contain NaN, Skip sending the action to the robot.")
            return
        self._publish_legs_cmd(robot_coordinates_action)

    """
    Functions that actually publish the commands and take effect
    """

    def _publish_legs_cmd(self, robot_coordinates_action: np.array):
        """Publish the joint commands to the robot legs in robot coordinates system.
        robot_coordinates_action: shape (NUM_JOINTS,), in simulation order.
        """
        for sim_idx in range(self.NUM_JOINTS):
            real_idx = self.joint_map[sim_idx]
            if not self.dryrun:
                self.low_cmd_buffer.motor_cmd[real_idx].mode = self.turn_on_motor_mode[sim_idx]
            self.low_cmd_buffer.motor_cmd[real_idx].q = (
                robot_coordinates_action[sim_idx] * self.joint_signs[sim_idx]
            ).item()
            self.low_cmd_buffer.motor_cmd[real_idx].dq = 0.0
            self.low_cmd_buffer.motor_cmd[real_idx].tau = 0.0
            self.low_cmd_buffer.motor_cmd[real_idx].kp = self.p_gains[sim_idx].item()
            self.low_cmd_buffer.motor_cmd[real_idx].kd = self.d_gains[sim_idx].item()

        self.low_cmd_buffer.crc = get_crc(self.low_cmd_buffer)
        if np.isnan(robot_coordinates_action).any():
            self.get_logger().error("Robot coordinates action contain NaN, Skip sending the action to the robot.")
            return
        self.low_cmd_publisher.publish(self.low_cmd_buffer)

    def _turn_off_motors(self):
        """Turn off the motors"""
        for sim_idx in range(self.NUM_JOINTS):
            real_idx = self.joint_map[sim_idx]
            self.low_cmd_buffer.motor_cmd[real_idx].mode = 0x00
            self.low_cmd_buffer.motor_cmd[real_idx].q = 0.0
            self.low_cmd_buffer.motor_cmd[real_idx].dq = 0.0
            self.low_cmd_buffer.motor_cmd[real_idx].tau = 0.0
            self.low_cmd_buffer.motor_cmd[real_idx].kp = 0.0
            self.low_cmd_buffer.motor_cmd[real_idx].kd = 0.0
        self.low_cmd_buffer.crc = get_crc(self.low_cmd_buffer)
        self.low_cmd_publisher.publish(self.low_cmd_buffer)
