import os, sys

import rclpy
from rclpy.node import Node
from unitree_hg.msg import (
    LowState,
    # MotorState,
    IMUState,
    LowCmd,
    # MotorCmd,
)
from unitree_go.msg import WirelessController
from std_msgs.msg import String, Float32MultiArray
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from crc_module import get_crc
import robot_cfgs

from multiprocessing import Process
from collections import OrderedDict
import numpy as np
import quaternion
# import torch
import re

# @torch.jit.script
# def quat_from_euler_xyz(roll, pitch, yaw):
#     """ roll, pitch, yaw in radians. quaterion in x, y, z, w order """
#     cy = torch.cos(yaw * 0.5)
#     sy = torch.sin(yaw * 0.5)
#     cr = torch.cos(roll * 0.5)
#     sr = torch.sin(roll * 0.5)
#     cp = torch.cos(pitch * 0.5)
#     sp = torch.sin(pitch * 0.5)

#     qw = cy * cr * cp + sy * sr * sp
#     qx = cy * sr * cp - sy * cr * sp
#     qy = cy * cr * sp + sy * sr * cp
#     qz = sy * cr * cp - cy * sr * sp

#     return torch.stack([qx, qy, qz, qw], dim=-1)

# @torch.jit.script
# def quat_rotate_inverse(q, v):
#     """ q must be in x, y, z, w order """
#     shape = q.shape
#     q_w = q[:, -1]
#     q_vec = q[:, :3]
#     a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
#     b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
#     c = q_vec * \
#         torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
#             shape[0], 3, 1)).squeeze(-1) * 2.0
#     return a - b + c

def quat_rotate_inverse(q: np.quaternion, v: np.array):
    """ q must be numpy-quaternion object in w, x, y, z order
    NOTE: non-batchwise version
    """
    q_inv = q.conjugate()
    return quaternion.rotate_vectors(q_inv, v)

class UnitreeRos2Real(Node):
    """ A proxy implementation of the real G1 robot.
    NOTE: different from go2 version, this class process all data in non-batchwise way.
    """
    class WirelessButtons:
        R1 =            0b00000001 # 1
        L1 =            0b00000010 # 2
        start =         0b00000100 # 4
        select =        0b00001000 # 8
        R2 =            0b00010000 # 16
        L2 =            0b00100000 # 32
        F1 =            0b01000000 # 64
        F2 =            0b10000000 # 128
        A =             0b100000000 # 256
        B =             0b1000000000 # 512
        X =             0b10000000000 # 1024
        Y =             0b100000000000 # 2048
        up =            0b1000000000000 # 4096
        right =         0b10000000000000 # 8192
        down =          0b100000000000000 # 16384
        left =          0b1000000000000000 # 32768

    def __init__(self,
            low_state_topic= "/lowstate",
            low_cmd_topic= "/lowcmd",
            imu_state_topic="/secondary_imu",
            joy_stick_topic= "/wirelesscontroller",
            cfg= dict(),
            lin_vel_deadband= 0.1,
            ang_vel_deadband= 0.1,
            cmd_px_range= [0.4, 1.0], # check joy_stick_callback (p for positive, n for negative)
            cmd_nx_range= [0.4, 0.8], # check joy_stick_callback (p for positive, n for negative)
            cmd_py_range= [0.4, 0.8], # check joy_stick_callback (p for positive, n for negative)
            cmd_ny_range= [0.4, 0.8], # check joy_stick_callback (p for positive, n for negative)
            cmd_pyaw_range= [0.4, 1.6], # check joy_stick_callback (p for positive, n for negative)
            cmd_nyaw_range= [0.4, 1.6], # check joy_stick_callback (p for positive, n for negative)
            move_by_wireless_remote= False, # if True, the robot will be controlled by a wireless remote
            computer_clip_torque= True, # if True, the actions will be clipped by torque limits
            joint_pos_protect_ratio= 1.5, # if the joint_pos is out of the range of this ratio, the process will shutdown.
            kp_factor= 1.0, # the factor to multiply the p_gain
            kd_factor= 1.0, # the factor to multiply the d_gain
            torque_limits_ratio= 1.0, # the factor to multiply the torque limits
            robot_class_name= "G1_29Dof",
            dryrun= True, # if True, the robot will not send commands to the real robot
        ):
        super().__init__("unitree_ros2_real")
        self.NUM_JOINTS = getattr(robot_cfgs, robot_class_name).NUM_JOINTS
        self.NUM_ACTIONS = getattr(robot_cfgs, robot_class_name).NUM_ACTIONS
        self.low_state_topic = low_state_topic
        self.imu_state_topic = imu_state_topic
        # Generate a unique cmd topic so that the low_cmd will not send to the robot's motor.
        self.low_cmd_topic = low_cmd_topic if not dryrun else low_cmd_topic + "_dryrun_" + str(np.random.randint(0, 65535))
        self.joy_stick_topic = joy_stick_topic
        self.cfg = cfg
        self.lin_vel_deadband = lin_vel_deadband
        self.ang_vel_deadband = ang_vel_deadband
        self.cmd_px_range = cmd_px_range
        self.cmd_nx_range = cmd_nx_range
        self.cmd_py_range = cmd_py_range
        self.cmd_ny_range = cmd_ny_range
        self.cmd_pyaw_range = cmd_pyaw_range
        self.cmd_nyaw_range = cmd_nyaw_range
        self.move_by_wireless_remote = move_by_wireless_remote
        self.computer_clip_torque = computer_clip_torque
        self.joint_pos_protect_ratio = joint_pos_protect_ratio
        self.kp_factor = kp_factor
        self.kd_factor = kd_factor
        self.torque_limits_ratio = torque_limits_ratio
        self.robot_class_name = robot_class_name
        self.dryrun = dryrun

        self.joint_map = getattr(robot_cfgs, robot_class_name).joint_map
        self.sim_joint_names = getattr(robot_cfgs, robot_class_name).sim_joint_names
        self.real_joint_names = getattr(robot_cfgs, robot_class_name).real_joint_names
        self.joint_signs = getattr(robot_cfgs, robot_class_name).joint_signs
        self.turn_on_motor_mode = getattr(robot_cfgs, robot_class_name).turn_on_motor_mode
        self.mode_pr = getattr(robot_cfgs, robot_class_name).mode_pr

        self.parse_config()

    def parse_config(self):
        """ Parse, set attributes from config dict, initialize buffers to speed up the computation """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.gravity_vec = np.zeros(3)
        self.gravity_vec[self.up_axis_idx] = -1
        
        # observations
        self.obs_clip = dict()
        self.obs_scales = dict()
        for obs_names, obs_config in self.cfg["observations"]["policy"].items():
            if obs_names == "concatenate_terms" or obs_names == "enable_corruption":
                continue
            if obs_config.get("clip", None) is not None:
                self.obs_clip[obs_names] = obs_config["clip"]
            if obs_config.get("scale", None) is not None:
                self.obs_scales[obs_names] = obs_config["scale"]

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
            assert "PDActuator" in actuator_config["class_type"], \
                "Only PDActuator trained model is supported for now. Get {} in actuator {}".format(actuator_config["class_type"], actuator_name)
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
        self.get_logger().info("PD gains are set to: p_gains: {}, d_gains: {}".format(self.p_gains, self.d_gains))
        self.get_logger().info("PD gains are set by kp_factor: {}, kd_factor: {}".format(self.kp_factor, self.kd_factor))
        self.torque_limits = getattr(robot_cfgs, self.robot_class_name).torque_limits * self.torque_limits_ratio
        self.get_logger().info("Torque limits are set by ratio of : {}".format(self.torque_limits_ratio))
        
        # buffers for observation output (in simulation order)
        self.joint_pos_ = np.zeros(self.NUM_JOINTS, dtype=np.float32) # in robot urdf coordinate, but in simulation order. no offset substracted
        self.joint_vel_ = np.zeros(self.NUM_JOINTS, dtype=np.float32)
        
        # actions
        self.actions_scale = np.zeros(self.NUM_ACTIONS, dtype=np.float32)
        for action_names, action_config in self.cfg["actions"].items():
            if not action_config["asset_name"] == "robot":
                continue
            for i in range(self.NUM_JOINTS):
                name = self.sim_joint_names[i]
                for _, joint_name_expr in enumerate(action_config["joint_names"]):
                    if re.search(joint_name_expr, name):
                        self.actions_scale[i] = action_config["scale"]
                        # print("Joint {}({}) has action scale {}".format(i, name, self.actions_scale[i]))
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
        """ After initializing the env and policy, register ros related callbacks and topics
        """

        # ROS publishers
        self.debug_msg_publisher = self.create_publisher(
            String,
            "/debug_msg",
            1
        )
        self.action_publisher = self.create_publisher(
            Float32MultiArray,
            "/raw_actions",
            1
        )
        self.low_cmd_pub = self.create_publisher(
            LowCmd,
            self.low_cmd_topic,
            1
        )
        self.low_cmd_buffer = LowCmd()
        self.low_cmd_buffer.mode_pr = self.mode_pr

        # ROS subscribers
        self.low_state_sub = self.create_subscription(
            LowState,
            self.low_state_topic,
            self._low_state_callback,
            1
        )
        self.torso_imu_sub = self.create_subscription(
            IMUState,
            self.imu_state_topic,
            self._torso_imu_state_callback,
            1
        )
        self.joy_stick_sub = self.create_subscription(
            WirelessController,
            self.joy_stick_topic,
            self._joy_stick_callback,
            1
        )

        self.debug_msg_publisher.publish(String(
            data= f"ROS handlers starting, kp: {self.p_gains}, kd: {self.d_gains}, torque_limits: {self.torque_limits}"
        ))
        self.debug_msg_publisher.publish(String(
            data= f"Using kp_factor: {self.kp_factor}, kd_factor: {self.kd_factor}, torque_limits_ratio: {self.torque_limits_ratio}"
        ))
        self.get_logger().info("ROS handlers started, waiting to recieve critical low state and wireless controller messages.")
        if not self.dryrun:
            self.get_logger().warn(f"You are running the code in no-dryrun mode and publishing to '{self.low_cmd_topic}', Please keep safe.")
        else:
            self.get_logger().warn(f"You are publishing low cmd to '{self.low_cmd_topic}' because of dryrun mode, Please check and be safe.")
        while rclpy.ok():
            rclpy.spin_once(self)
            if hasattr(self, "low_state_buffer") and hasattr(self, "joy_stick_buffer"):
                break
        self.get_logger().info("Low state message received, the robot is ready to go.")

    """
    ROS callbacks and handlers that update the buffer
    """

    def _low_state_callback(self, msg):
        """ store and handle proprioception data """
        self.low_state_buffer = msg # keep the latest low state
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
            if self.joint_pos_[sim_idx] > self.joint_pos_protect_high[sim_idx] or \
                self.joint_pos_[sim_idx] < self.joint_pos_protect_low[sim_idx]:
                self.get_logger().error(f"Joint {sim_idx}(sim), {real_idx}(real) position out of range at {self.low_state_buffer.motor_state[real_idx].q}")
                self.debug_msg_publisher.publish(String(data= f"Joint {sim_idx}(sim), {real_idx}(real) position out of range at {self.low_state_buffer.motor_state[real_idx].q}"))
                self.get_logger().error("The motors and this process shuts down.")
                self._turn_off_motors()
                raise SystemExit()
            
    def _torso_imu_state_callback(self, msg):
        """ store and handle torso imu data """
        self.get_logger().info("Torso IMU data received.", once=True)
        self.torso_imu_buffer = msg

    def _joy_stick_callback(self, msg):
        self.joy_stick_buffer = msg
        if self.move_by_wireless_remote:
            # left-y for forward/backward
            ly = msg.ly
            if ly > self.lin_vel_deadband:
                vx = (ly - self.lin_vel_deadband) / (1 - self.lin_vel_deadband) # (0, 1)
                vx = vx * (self.cmd_px_range[1] - self.cmd_px_range[0]) + self.cmd_px_range[0]
            elif ly < -self.lin_vel_deadband:
                vx = (ly + self.lin_vel_deadband) / (1 - self.lin_vel_deadband) # (-1, 0)
                vx = vx * (self.cmd_nx_range[1] - self.cmd_nx_range[0]) - self.cmd_nx_range[0]
            else:
                vx = 0
            # left-x for turning left/right
            lx = -msg.lx
            if lx > self.ang_vel_deadband:
                yaw = (lx - self.ang_vel_deadband) / (1 - self.ang_vel_deadband)
                yaw = yaw * (self.cmd_pyaw_range[1] - self.cmd_pyaw_range[0]) + self.cmd_pyaw_range[0]
            elif lx < -self.ang_vel_deadband:
                yaw = (lx + self.ang_vel_deadband) / (1 - self.ang_vel_deadband)
                yaw = yaw * (self.cmd_nyaw_range[1] - self.cmd_nyaw_range[0]) - self.cmd_nyaw_range[0]
            else:
                yaw = 0
            # right-x for side moving left/right
            rx = -msg.rx
            if rx > self.lin_vel_deadband:
                vy = (rx - self.lin_vel_deadband) / (1 - self.lin_vel_deadband)
                vy = vy * (self.cmd_py_range[1] - self.cmd_py_range[0]) + self.cmd_py_range[0]
            elif rx < -self.lin_vel_deadband:
                vy = (rx + self.lin_vel_deadband) / (1 - self.lin_vel_deadband)
                vy = vy * (self.cmd_ny_range[1] - self.cmd_ny_range[0]) - self.cmd_ny_range[0]
            else:
                vy = 0
            self.xyyaw_command = np.array([vx, vy, yaw], dtype=np.float32)

        # refer to Unitree Remote Control data structure, msg.keys is a bit mask
        # 00000000 00000001 means pressing the 0-th button (R1)
        # 00000000 00000010 means pressing the 1-th button (L1)
        # 10000000 00000000 means pressing the 15-th button (left)
        if (msg.keys & self.WirelessButtons.R2) or (msg.keys & self.WirelessButtons.L2): # R2 or L2 is pressed
            self.get_logger().warn("R2 or L2 is pressed, the motors and this process shuts down.")
            self._turn_off_motors()
            raise SystemExit()

    def _forward_depth_callback(self, msg):
        """ store and handle depth camera data """
        pass

    def _forward_depth_embedding_callback(self, msg):
        self.forward_depth_embedding_buffer = np.array(msg.data)

    """
    Refresh observation buffer and corresponding sub-functions
    """

    def _get_lin_vel_obs(self):
        return np.zeros(3, dtype=np.float32)
    
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
        return quat_rotate_inverse(
            quat_wxyz,
            self.gravity_vec,
        ).astype(np.float32)

    def _get_commands_obs(self):
        return self.xyyaw_command

    def _get_joint_pos_obs(self):
        return self.joint_pos_ - self.default_joint_pos

    def _get_joint_vel_obs(self):
        return self.joint_vel_

    def _get_last_actions_obs(self):
        return self.actions
    
    def get_num_obs_from_components(self, components):
        obs_segments = self.get_obs_segment_from_components(components)
        num_obs = 0
        for k, v in obs_segments.items():
            num_obs += np.prod(v)
        return num_obs
        
    def get_obs_segment_from_components(self, components):
        """ Observation segment is defined as a list of lists/ints defining the tensor shape with
        corresponding order.
        """
        segments = OrderedDict()
        if "lin_vel" in components:
            print("Warning: lin_vel is not typically available or accurate enough on the real robot. Will return zeros.")
            segments["lin_vel"] = (3,)
        if "ang_vel" in components:
            segments["ang_vel"] = (3,)
        if "projected_gravity" in components:
            segments["projected_gravity"] = (3,)
        if "commands" in components:
            segments["commands"] = (3,)
        if "joint_pos" in components:
            segments["joint_pos"] = (self.NUM_JOINTS,)
        if "joint_vel" in components:
            segments["joint_vel"] = (self.NUM_JOINTS,)
        if "last_actions" in components:
            segments["last_actions"] = (self.NUM_ACTIONS,)
        
        return segments
    
    def get_obs(self, obs_segments= None):
        """ Extract from the buffers and build the 1d observation tensor
        Each get ... obs function does not do the obs_scale multiplication.
        NOTE: obs_buffer has the batch dimension, whose size is 1.
        """
        if obs_segments is None:
            obs_segments = self.obs_segments
        obs_buffer = []
        for k, v in obs_segments.items():
            obs_component_value = getattr(self, "_get_" + k + "_obs")() * self.obs_scales.get(k, 1.0)
            obs_buffer.append(obs_component_value)
        obs_buffer = np.concatenate(obs_buffer)
        obs_buffer = np.clip(obs_buffer, -self.obs_clip, self.obs_clip)
        return obs_buffer

    """
    Control related functions
    """
    def clip_by_torque_limit(self, actions_scaled):
        """ Different from simulation, we reverse the process and clip the actions directly,
        so that the PD controller runs in robot but not our script.
        """

        # NOTE: Currently only support position control with PD controller
        p_limits_low = (-self.torque_limits) + self.d_gains*self.joint_vel_
        p_limits_high = (self.torque_limits) + self.d_gains*self.joint_vel_
        actions_low = (p_limits_low/self.p_gains) - self.default_joint_pos + self.joint_pos_
        actions_high = (p_limits_high/self.p_gains) - self.default_joint_pos + self.joint_pos_

        return np.clip(actions_scaled, actions_low, actions_high)

    def send_action(self, actions):
        """ Send the action to the robot motors, which does the preprocessing
        just like env.step in simulation.
        Thus, the actions has the batch dimension, whose size is 1.
        """
        self.action_publisher.publish(Float32MultiArray(data= actions))
        if self.computer_clip_torque:
            clipped_scaled_action = self.clip_by_torque_limit(actions * self.actions_scale)
            clipped_joints = np.where(clipped_scaled_action != actions * self.actions_scale)[0]
            if len(clipped_joints) > 0:
                self.get_logger().warn(f"Computer Clip Torque is True, the following joints are clipped: {clipped_joints}", throttle_duration_sec= 5)
                self.debug_msg_publisher.publish(String(data= f"Computer Clip Torque is True, the following joints are clipped: {clipped_joints}"))
        else:
            self.get_logger().warn("Computer Clip Torque is False, the robot may be damaged.", throttle_duration_sec= 5)
            clipped_scaled_action = actions * self.actions_scale
        robot_coordinates_action = clipped_scaled_action + self.default_joint_pos
        if np.isnan(actions).any():
            self.get_logger().error("Actions contain NaN, Skip sending the action to the robot.")
            return
        self._publish_legs_cmd(robot_coordinates_action)

    """
    functions that actually publish the commands and take effect
    """

    def _publish_legs_cmd(self, robot_coordinates_action: np.array):
        """ Publish the joint commands to the robot legs in robot coordinates system.
        robot_coordinates_action: shape (NUM_JOINTS,), in simulation order.
        """
        for sim_idx in range(self.NUM_JOINTS):
            real_idx = self.joint_map[sim_idx]
            if not self.dryrun:
                self.low_cmd_buffer.motor_cmd[real_idx].mode = self.turn_on_motor_mode[sim_idx]
            self.low_cmd_buffer.motor_cmd[real_idx].q = (robot_coordinates_action[sim_idx] * self.joint_signs[sim_idx]).item()
            self.low_cmd_buffer.motor_cmd[real_idx].dq = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].tau = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].kp = self.p_gains[sim_idx].item()
            self.low_cmd_buffer.motor_cmd[real_idx].kd = self.d_gains[sim_idx].item()
        
        self.low_cmd_buffer.crc = get_crc(self.low_cmd_buffer)
        if np.isnan(robot_coordinates_action).any():
            self.get_logger().error("Robot coordinates action contain NaN, Skip sending the action to the robot.", throttle_duration_sec= 2)
            self.debug_msg_publisher.publish(String(data= "Robot coordinates action contain NaN, Skip sending the action to the robot."))
            return
        self.low_cmd_pub.publish(self.low_cmd_buffer)

    def _turn_off_motors(self):
        """ Turn off the motors """
        for sim_idx in range(self.NUM_JOINTS):
            real_idx = self.joint_map[sim_idx]
            self.low_cmd_buffer.motor_cmd[real_idx].mode = 0x00
            self.low_cmd_buffer.motor_cmd[real_idx].q = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].dq = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].tau = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].kp = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].kd = 0.
        self.low_cmd_buffer.crc = get_crc(self.low_cmd_buffer)
        self.low_cmd_pub.publish(self.low_cmd_buffer)
