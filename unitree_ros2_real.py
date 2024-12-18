import os, sys

import rclpy
from rclpy.node import Node
from unitree_hg.msg import (
    LowState,
    # MotorState,
    # IMUState,
    LowCmd,
    # MotorCmd,
)
from unitree_go.msg import WirelessController
from std_msgs.msg import Float32MultiArray
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from crc_module import get_crc

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

class RobotCfgs:
    class G1_29Dof:
        NUM_DOF = 29
        NUM_ACTIONS = 29
        dof_map = [ # dof_map[sim_idx] == real_idx
            # You may check the consistency with `sim_dof_names` and `real_dof_names`
            15, 22, # shoulder pitch
            14, # waist pitch
            16, 23, # shoulder roll
            13, # waist roll
            17, 24, # shoulder yaw
            12, # waist yaw
            18, 25, # elbow
            0, 6, # hip pitch
            19, 26, # wrist roll
            1, 7, # hip roll
            20, 27, # wrist pitch
            2, 8, # hip yaw
            21, 28, # wrist yaw
            3, 9, # knee
            4, 10, # ankle pitch
            5, 11, # ankle roll
        ]
        sim_dof_names = [ # NOTE: order matters. This list is the order in simulation.
            'left_shoulder_pitch_joint', #
            'right_shoulder_pitch_joint',
            'waist_pitch_joint',
            'left_shoulder_roll_joint', #
            'right_shoulder_roll_joint',
            'waist_roll_joint',
            'left_shoulder_yaw_joint', #
            'right_shoulder_yaw_joint',
            'waist_yaw_joint',
            'left_elbow_joint', #
            'right_elbow_joint',
            'left_hip_pitch_joint',
            'right_hip_pitch_joint',
            'left_wrist_roll_joint',
            'right_wrist_roll_joint',
            'left_hip_roll_joint', #
            'right_hip_roll_joint',
            'left_wrist_pitch_joint',
            'right_wrist_pitch_joint',
            'left_hip_yaw_joint',
            'right_hip_yaw_joint',
            'left_wrist_yaw_joint', #
            'right_wrist_yaw_joint',
            'left_knee_joint',
            'right_knee_joint',
            'left_ankle_pitch_joint', #
            'right_ankle_pitch_joint',
            'left_ankle_roll_joint',
            'right_ankle_roll_joint',
        ]
        real_dof_names = [ # NOTE: order matters. This list is the order in real robot.
            'left_hip_pitch_joint',
            'left_hip_roll_joint',
            'left_hip_yaw_joint',
            'left_knee_joint',
            'left_ankle_pitch_joint',
            'left_ankle_roll_joint',
            'right_hip_pitch_joint',
            'right_hip_roll_joint',
            'right_hip_yaw_joint',
            'right_knee_joint',
            'right_ankle_pitch_joint',
            'right_ankle_roll_joint',
            'waist_yaw_joint',
            'waist_roll_joint',
            'waist_pitch_joint',
            'left_shoulder_pitch_joint',
            'left_shoulder_roll_joint',
            'left_shoulder_yaw_joint',
            'left_elbow_joint',
            'left_wrist_roll_joint',
            'left_wrist_pitch_joint',
            'left_wrist_yaw_joint',
            'right_shoulder_pitch_joint',
            'right_shoulder_roll_joint',
            'right_shoulder_yaw_joint',
            'right_elbow_joint',
            'right_wrist_roll_joint',
            'right_wrist_pitch_joint',
            'right_wrist_yaw_joint',
        ]
        dof_signs = [ # in simulation order
            1, 1, -1,
            1, 1, -1,
            1, 1, -1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ]
        joint_limits_high = np.array([ # in simulation order
            2.6704, 2.6704, 0.5200,
            2.2515, 1.5882, 0.5200,
            2.6180, 2.6180, 2.6180,
            2.0944, 2.0944, 2.8798, 2.8798, 1.9722, 1.9722,
            2.9671, 0.5236, 1.6144, 1.6144, 2.7576, 2.7576,
            1.6144, 1.6144, 2.8798, 2.8798,
            0.5236, 0.5236, 0.2618, 0.2618,
        ])
        joint_limits_low = np.array([ # in simulation order
            -3.0892, -3.0892, -0.5200,
            -1.5882, -2.2515, -0.5200,
            -2.6180, -2.6180, -2.6180,
            -1.0472, -1.0472, -2.5307, -2.5307, -1.9722, -1.9722,
            -0.5236, -2.9671, -1.6144, -1.6144, -2.7576, -2.7576,
            -1.6144, -1.6144, -0.0873, -0.0873,
            -0.8727, -0.8727, -0.2618, -0.2618,
        ])
        torque_limits = np.array([ # in simulation order
            25, 25, 50,
            25, 25, 50,
            25, 25, 88,
            25, 25, 88, 88, 25, 25,
            88, 88, 5, 5, 88, 88,
            5, 5, 139, 139,
            50, 50, 50, 50,
        ])
        turn_on_motor_mode = [0x01] * 29
        mode_pr = 0
        """ please check this value from
            https://support.unitree.com/home/zh/G1_developer/basic_services_interface
            https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description
        """

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
            dof_pos_protect_ratio= 1.01, # if the dof_pos is out of the range of this ratio, the process will shutdown.
            robot_class_name= "G1_29Dof",
            dryrun= True, # if True, the robot will not send commands to the real robot
        ):
        super().__init__("unitree_ros2_real")
        self.NUM_DOF = getattr(RobotCfgs, robot_class_name).NUM_DOF
        self.NUM_ACTIONS = getattr(RobotCfgs, robot_class_name).NUM_ACTIONS
        self.low_state_topic = low_state_topic
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
        self.dof_pos_protect_ratio = dof_pos_protect_ratio
        self.robot_class_name = robot_class_name
        self.dryrun = dryrun

        self.dof_map = getattr(RobotCfgs, robot_class_name).dof_map
        self.sim_dof_names = getattr(RobotCfgs, robot_class_name).sim_dof_names
        self.real_dof_names = getattr(RobotCfgs, robot_class_name).real_dof_names
        self.dof_signs = getattr(RobotCfgs, robot_class_name).dof_signs
        self.turn_on_motor_mode = getattr(RobotCfgs, robot_class_name).turn_on_motor_mode
        self.mode_pr = getattr(RobotCfgs, robot_class_name).mode_pr

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
        self.default_dof_pos = np.zeros(self.NUM_DOF, dtype=np.float32)
        for joint_name_expr, joint_pos in self.cfg["scene"]["robot"]["init_state"]["joint_pos"].items():
            # compute the default dof pos from configuration for articulation.default_dof_pos
            for i in range(self.NUM_DOF):
                name = self.sim_dof_names[i]
                if re.search(joint_name_expr, name):
                    self.default_dof_pos[i] = joint_pos
        self.p_gains = np.zeros(self.NUM_DOF, dtype=np.float32)
        self.d_gains = np.zeros(self.NUM_DOF, dtype=np.float32)
        for actuator_name, actuator_config in self.cfg["scene"]["robot"]["actuators"].items():
            assert "PDActuator" in actuator_config["class_type"], \
                "Only PDActuator trained model is supported for now. Get {} in actuator {}".format(actuator_config["class_type"], actuator_name)
            for i in range(self.NUM_DOF):
                name = self.sim_dof_names[i]
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
                        # print(f"Joint {i}({self.sim_dof_names[i]}) has p_gain {self.p_gains[i]} and d_gain {self.d_gains[i]}")
        self.torque_limits = getattr(RobotCfgs, self.robot_class_name).torque_limits
        
        # buffers for observation output (in simulation order)
        self.dof_pos_ = np.zeros(self.NUM_DOF, dtype=np.float32) # in robot urdf coordinate, but in simulation order. no offset substracted
        self.dof_vel_ = np.zeros(self.NUM_DOF, dtype=np.float32)
        
        # actions
        self.actions_scale = np.zeros(self.NUM_ACTIONS, dtype=np.float32)
        for action_names, action_config in self.cfg["actions"].items():
            if not action_config["asset_name"] == "robot":
                continue
            for i in range(self.NUM_DOF):
                name = self.sim_dof_names[i]
                for _, joint_name_expr in enumerate(action_config["joint_names"]):
                    if re.search(joint_name_expr, name):
                        self.actions_scale[i] = action_config["scale"]
                        # print("Joint {}({}) has action scale {}".format(i, name, self.actions_scale[i]))
                    if not action_config["use_default_offset"]:
                        # not using articulation.default_dof_pos as default offset
                        if isinstance(action_config["offset"], dict):
                            self.default_dof_pos[i] = action_config["offset"][joint_name_expr]
                        else:
                            self.default_dof_pos[i] = action_config["offset"]            
        self.actions_raw = np.zeros(self.NUM_ACTIONS, dtype=np.float32)
        self.actions = np.zeros(self.NUM_ACTIONS, dtype=np.float32)

        # hardware related, in simulation order
        self.joint_limits_high = getattr(RobotCfgs, self.robot_class_name).joint_limits_high
        self.joint_limits_low = getattr(RobotCfgs, self.robot_class_name).joint_limits_low
        joint_pos_mid = (self.joint_limits_high + self.joint_limits_low) / 2
        joint_pos_range = (self.joint_limits_high - self.joint_limits_low) / 2
        self.joint_pos_protect_high = joint_pos_mid + joint_pos_range * self.dof_pos_protect_ratio
        self.joint_pos_protect_low = joint_pos_mid - joint_pos_range * self.dof_pos_protect_ratio

    def start_ros_handlers(self):
        """ After initializing the env and policy, register ros related callbacks and topics
        """

        # ROS publishers
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
        self.joy_stick_sub = self.create_subscription(
            WirelessController,
            self.joy_stick_topic,
            self._joy_stick_callback,
            1
        )

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

        # refresh dof_pos and dof_vel
        for sim_idx in range(self.NUM_DOF):
            real_idx = self.dof_map[sim_idx]
            self.dof_pos_[sim_idx] = self.low_state_buffer.motor_state[real_idx].q * self.dof_signs[sim_idx]
        for sim_idx in range(self.NUM_DOF):
            real_idx = self.dof_map[sim_idx]
            self.dof_vel_[sim_idx] = self.low_state_buffer.motor_state[real_idx].dq * self.dof_signs[sim_idx]
        # automatic safety check
        for sim_idx in range(self.NUM_DOF):
            real_idx = self.dof_map[sim_idx]
            if self.dof_pos_[sim_idx] > self.joint_pos_protect_high[sim_idx] or \
                self.dof_pos_[sim_idx] < self.joint_pos_protect_low[sim_idx]:
                self.get_logger().error(f"Joint {sim_idx}(sim), {real_idx}(real) position out of range at {self.low_state_buffer.motor_state[real_idx].q}")
                self.get_logger().error("The motors and this process shuts down.")
                self._turn_off_motors()
                raise SystemExit()

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
        return np.array(self.low_state_buffer.imu_state.gyroscope, dtype=np.float32)

    def _get_projected_gravity_obs(self):
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

    def _get_dof_pos_obs(self):
        return self.dof_pos_ - self.default_dof_pos

    def _get_dof_vel_obs(self):
        return self.dof_vel_

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
        if "dof_pos" in components:
            segments["dof_pos"] = (self.NUM_DOF,)
        if "dof_vel" in components:
            segments["dof_vel"] = (self.NUM_DOF,)
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
        p_limits_low = (-self.torque_limits) + self.d_gains*self.dof_vel_
        p_limits_high = (self.torque_limits) + self.d_gains*self.dof_vel_
        actions_low = (p_limits_low/self.p_gains) - self.default_dof_pos + self.dof_pos_
        actions_high = (p_limits_high/self.p_gains) - self.default_dof_pos + self.dof_pos_

        return np.clip(actions_scaled, actions_low, actions_high)

    def send_action(self, actions):
        """ Send the action to the robot motors, which does the preprocessing
        just like env.step in simulation.
        Thus, the actions has the batch dimension, whose size is 1.
        """
        if self.computer_clip_torque:
            clipped_scaled_action = self.clip_by_torque_limit(actions * self.actions_scale)
        else:
            self.get_logger().warn("Computer Clip Torque is False, the robot may be damaged.", throttle_duration_sec= 5)
            clipped_scaled_action = actions * self.actions_scale
        robot_coordinates_action = clipped_scaled_action + self.default_dof_pos

        self._publish_legs_cmd(robot_coordinates_action)

    """
    functions that actually publish the commands and take effect
    """

    def _publish_legs_cmd(self, robot_coordinates_action: np.array):
        """ Publish the joint commands to the robot legs in robot coordinates system.
        robot_coordinates_action: shape (NUM_DOF,), in simulation order.
        """
        for sim_idx in range(self.NUM_DOF):
            real_idx = self.dof_map[sim_idx]
            if not self.dryrun:
                self.low_cmd_buffer.motor_cmd[real_idx].mode = self.turn_on_motor_mode[sim_idx]
            self.low_cmd_buffer.motor_cmd[real_idx].q = robot_coordinates_action[sim_idx].item() * self.dof_signs[sim_idx]
            self.low_cmd_buffer.motor_cmd[real_idx].dq = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].tau = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].kp = self.p_gains[sim_idx].item()
            self.low_cmd_buffer.motor_cmd[real_idx].kd = self.d_gains[sim_idx].item()
        
        self.low_cmd_buffer.crc = get_crc(self.low_cmd_buffer)
        self.low_cmd_pub.publish(self.low_cmd_buffer)

    def _turn_off_motors(self):
        """ Turn off the motors """
        for sim_idx in range(self.NUM_DOF):
            real_idx = self.dof_map[sim_idx]
            self.low_cmd_buffer.motor_cmd[real_idx].mode = 0x00
            self.low_cmd_buffer.motor_cmd[real_idx].q = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].dq = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].tau = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].kp = 0.
            self.low_cmd_buffer.motor_cmd[real_idx].kd = 0.
        self.low_cmd_buffer.crc = get_crc(self.low_cmd_buffer)
        self.low_cmd_pub.publish(self.low_cmd_buffer)
