import re

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
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

        self.torque_limits = getattr(robot_cfgs, self.robot_class_name).torque_limits * self.torque_limits_ratio
        self.get_logger().info(f"Torque limits are set by ratio of : {self.torque_limits_ratio}")

        # buffers for observation output (in simulation order)
        self.joint_pos_ = np.zeros(
            self.NUM_JOINTS, dtype=np.float32
        )  # in robot urdf coordinate, but in simulation order. no offset subtracted
        self.joint_vel_ = np.zeros(self.NUM_JOINTS, dtype=np.float32)

        # action (for get_last_action_obs across multiple agents)
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
        self.action_publisher = self.create_publisher(Float32MultiArray, "/raw_actions", 10)
        self.low_cmd_publisher = self.create_publisher(LowCmd, self.low_cmd_topic, 10)
        self.low_cmd_buffer = LowCmd()
        self.low_cmd_buffer.mode_pr = self.mode_pr

        # ROS subscribers
        self.low_state_subscriber = self.create_subscription(
            LowState, self.low_state_topic, self._low_state_callback, 10
        )
        self.torso_imu_subscriber = self.create_subscription(
            IMUState, self.imu_state_topic, self._torso_imu_state_callback, 10
        )
        self.joy_stick_subscriber = self.create_subscription(
            WirelessController, self.joy_stick_topic, self._joy_stick_callback, 10
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

    def _get_base_ang_vel_obs(self):
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

    def _get_joint_vel_obs(self):
        return self.joint_vel_

    def _get_last_action_obs(self):
        return self.actions

    """
    Control related functions
    """

    def clip_by_torque_limit(
        self,
        target_joint_pos,
        p_gains: np.ndarray = 0.0,
        d_gains: np.ndarray = 0.0,
    ):
        """Different from simulation, we reverse the process and clip the target position directly,
        so that the PD controller runs in robot but not our script.
        """
        p_limits_low = (-self.torque_limits) + d_gains * self.joint_vel_
        p_limits_high = (self.torque_limits) + d_gains * self.joint_vel_
        action_low = (p_limits_low / p_gains) + self.joint_pos_
        action_high = (p_limits_high / p_gains) + self.joint_pos_

        return np.clip(target_joint_pos, action_low, action_high)

    def send_action(
        self,
        action: np.array,
        action_offset: np.array = 0.0,
        action_scale: np.ndarray = 1.0,
        p_gains: np.ndarray = 0.0,
        d_gains: np.ndarray = 0.0,
    ):
        """Send the action to the robot motors, which does the preprocessing
        just like env.step in simulation.
        However, since this process only controls one robot, the action is not batched.
        NOTE: when switching between agents, the last_action term should be shared between agents.
        Thus, the ros node has to update the action buffer
        """
        # NOTE: Only calling this function currently will update self.actions for self._get_last_action_obs
        self.actions[:] = action
        self.action_publisher.publish(Float32MultiArray(data=action))
        action_scaled = action * action_scale
        target_joint_pos = action_scaled + action_offset
        p_gains = np.clip(p_gains * self.kp_factor, 0.0, 500.0)
        d_gains = np.clip(d_gains * self.kd_factor, 0.0, 5.0)
        if np.isnan(action).any():
            self.get_logger().error("Actions contain NaN, Skip sending the action to the robot.")
            return
        if self.computer_clip_torque:
            target_joint_pos = self.clip_by_torque_limit(
                target_joint_pos,
                p_gains=p_gains,
                d_gains=d_gains,
            )
        # target_joint_pos=np.clip(a=target_joint_pos, a_max=self.joint_limits_high, a_min=self.joint_limits_low)
        self._publish_motor_cmd(target_joint_pos, p_gains=p_gains, d_gains=d_gains)

    """
    Functions that actually publish the commands and take effect
    """

    def _publish_motor_cmd(
        self,
        target_joint_pos: np.array,  # shape (NUM_JOINTS,), in simulation order
        p_gains: np.ndarray,  # In the order of simulation joints, not real joints
        d_gains: np.ndarray,  # In the order of simulation joints, not real joints
    ):
        """Publish the joint commands to the robot motors in robot coordinates system.
        robot_coordinates_action: shape (NUM_JOINTS,), in simulation order.
        """
        for sim_idx in range(self.NUM_JOINTS):
            real_idx = self.joint_map[sim_idx]
            if not self.dryrun:
                self.low_cmd_buffer.motor_cmd[real_idx].mode = self.turn_on_motor_mode[sim_idx]
            self.low_cmd_buffer.motor_cmd[real_idx].q = (target_joint_pos[sim_idx] * self.joint_signs[sim_idx]).item()
            self.low_cmd_buffer.motor_cmd[real_idx].dq = 0.0
            self.low_cmd_buffer.motor_cmd[real_idx].tau = 0.0
            self.low_cmd_buffer.motor_cmd[real_idx].kp = p_gains[sim_idx].item()
            self.low_cmd_buffer.motor_cmd[real_idx].kd = d_gains[sim_idx].item()

        self.low_cmd_buffer.crc = get_crc(self.low_cmd_buffer)
        if np.isnan(target_joint_pos).any():
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
