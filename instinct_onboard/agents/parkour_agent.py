from __future__ import annotations

import os
import re
from typing import Tuple

import numpy as np
import onnxruntime as ort
import prettytable
import yaml

from instinct_onboard.agents.base import OnboardAgent
from instinct_onboard.ros_nodes.ros_real import Ros2Real


class ParkourAgent(OnboardAgent):
    def __init__(
        self,
        logdir: str,
        ros_node: Ros2Real,
    ):
        super().__init__(logdir, ros_node)
        self.ort_sessions = dict()
        self.lin_vel_deadband = 0.15
        self.ang_vel_deadband = 0.15
        self.cmd_px_range = [0.0, 1.0]
        self.cmd_nx_range = [0.0, 0.0]
        self.cmd_py_range = [0.0, 0.3]
        self.cmd_ny_range = [0.0, 0.3]
        self.cmd_pyaw_range = [0.0, 1.0]
        self.cmd_nyaw_range = [0.0, 1.0]
        self.move_by_wireless_remote = True
        self._parse_obs_config()
        self._parse_action_config()
        self._load_models()

    def _parse_obs_config(self):
        super()._parse_obs_config()
        with open(os.path.join(self.logdir, "params", "agent.yaml")) as f:
            self.agent_cfg = yaml.unsafe_load(f)
        all_obs_names = list(self.obs_funcs.keys())
        self.proprio_obs_names = [obs_name for obs_name in all_obs_names]
        self.ros_node.get_logger().info(f"ParkourAgent proprioception names: {self.proprio_obs_names}")
        table = prettytable.PrettyTable()
        table.field_names = ["Observation Name", "Function"]
        for obs_name, func in self.obs_funcs.items():
            table.add_row([obs_name, func.__name__])
        print("Observation functions:")
        print(table)

    def _parse_action_config(self):
        super()._parse_action_config()
        self._zero_action_joints = np.zeros(self.ros_node.NUM_ACTIONS, dtype=np.float32)
        for action_names, action_config in self.cfg["actions"].items():
            for i in range(self.ros_node.NUM_JOINTS):
                name = self.ros_node.sim_joint_names[i]
                for _, joint_name_expr in enumerate(action_config["default_joint_names"]):
                    if re.search(joint_name_expr, name):
                        self._zero_action_joints[i] = 1.0

    def _parse_observation_function(self, obs_name, obs_config):
        obs_func = obs_config["func"].split(":")[-1]  # get the function name from the config
        if obs_func == "depth_image":
            obs_name = "depth_latent"
            if hasattr(self, f"_get_{obs_name}_obs"):
                self.obs_funcs[obs_name] = getattr(self, f"_get_{obs_name}_obs")
                return
            else:
                raise ValueError(f"Unknown observation function for observation {obs_name}")
        return super()._parse_observation_function(obs_name, obs_config)

    def _load_models(self):
        """Load the ONNX model for the agent."""
        # load ONNX models
        ort_execution_providers = ort.get_available_providers()
        actor_path = os.path.join(self.logdir, "exported", "actor.onnx")
        self.ort_sessions["actor"] = ort.InferenceSession(actor_path, providers=ort_execution_providers)
        print(f"Loaded ONNX models from {self.logdir}")

    def reset(self):
        """Reset the agent state and the rosbag reader."""
        pass

    def step(self):
        """Perform a single step of the agent."""
        # pack actor MLP input
        proprio_obs = []
        for proprio_obs_name in self.proprio_obs_names:
            obs_term_value = self._get_single_obs_term(proprio_obs_name)
            proprio_obs.append(np.reshape(obs_term_value, (1, -1)).astype(np.float32))
        proprio_obs = np.concatenate(proprio_obs, axis=-1)

        # run actor MLP
        actor_input_name = self.ort_sessions["actor"].get_inputs()[0].name
        action = self.ort_sessions["actor"].run(None, {actor_input_name: proprio_obs})[0]
        action = action.reshape(-1)
        # reconstruct full action including zeroed joints
        mask = (self._zero_action_joints == 0).astype(bool)
        full_action = np.zeros(self.ros_node.NUM_ACTIONS, dtype=np.float32)
        full_action[mask] = action

        done = False

        return full_action, done

    """
    Agent specific observation functions for Parkour Agent.
    """

    def _get_base_lin_vel_zero_obs(self):
        """Return shape: (3,)"""
        return np.zeros((3,))

    def _get_base_velocity_obs(self):
        """Return shape: (3,)"""
        if self.move_by_wireless_remote:
            joy_stick_command = self.ros_node.joy_stick_command  # [Lx, Ly, Rx, Ry]
            # left-y for forward/backward
            ly = joy_stick_command[1]
            if ly > self.lin_vel_deadband:
                vx = (ly - self.lin_vel_deadband) / (1 - self.lin_vel_deadband)  # (0, 1)
                vx = vx * (self.cmd_px_range[1] - self.cmd_px_range[0]) + self.cmd_px_range[0]
            elif ly < -self.lin_vel_deadband:
                vx = (ly + self.lin_vel_deadband) / (1 - self.lin_vel_deadband)  # (-1, 0)
                vx = vx * (self.cmd_nx_range[1] - self.cmd_nx_range[0]) - self.cmd_nx_range[0]
            else:
                vx = 0
            # left-x for side moving left/right
            lx = -joy_stick_command[0]
            if lx > self.lin_vel_deadband:
                vy = (lx - self.lin_vel_deadband) / (1 - self.lin_vel_deadband)
                vy = vy * (self.cmd_py_range[1] - self.cmd_py_range[0]) + self.cmd_py_range[0]
            elif lx < -self.lin_vel_deadband:
                vy = (lx + self.lin_vel_deadband) / (1 - self.lin_vel_deadband)
                vy = vy * (self.cmd_ny_range[1] - self.cmd_ny_range[0]) - self.cmd_ny_range[0]
            else:
                vy = 0
            # right-x for turning left/right
            rx = -joy_stick_command[2]
            if rx > self.ang_vel_deadband:
                yaw = (rx - self.ang_vel_deadband) / (1 - self.ang_vel_deadband)
                yaw = yaw * (self.cmd_pyaw_range[1] - self.cmd_pyaw_range[0]) + self.cmd_pyaw_range[0]
            elif rx < -self.ang_vel_deadband:
                yaw = (rx + self.ang_vel_deadband) / (1 - self.ang_vel_deadband)
                yaw = yaw * (self.cmd_nyaw_range[1] - self.cmd_nyaw_range[0]) - self.cmd_nyaw_range[0]
            else:
                yaw = 0

            self.xyyaw_command = np.array([vx, vy, yaw], dtype=np.float32)
            return self.xyyaw_command

    def _get_joint_vel_rel_obs(self):
        """Return shape: (num_joints,)"""
        return self.ros_node.joint_vel_

    def _get_depth_latent_obs(self):
        """Return shape: (depth_latent_dim,)"""
        return self.ros_node.depth_latent_buffer

    def _get_last_action_obs(self):
        """Return shape: (num_active_joints,)"""
        actions = np.asarray(self.ros_node.actions).astype(np.float32)
        mask = (1.0 - self._zero_action_joints).astype(bool)
        return actions[mask]


class ParkourColdStartAgent(ParkourAgent):
    def __init__(
        self, logdir: str, dof_max_err: float, start_steps: int, ros_node: Ros2Real, joint_target_pos: np.array = None
    ):
        """Initialize the parkour cold start agent.
        Args:
            startup_step_size (float): The step size for the cold start agent to move the joints.
            ros_node (Ros2Real): The ROS node instance to interact with the robot.
        """
        super().__init__(logdir, ros_node)
        self.ros_node = ros_node
        self.dof_max_err = dof_max_err
        self.start_steps = start_steps
        self.joint_target_pos = np.zeros(self.ros_node.NUM_JOINTS) if joint_target_pos is None else joint_target_pos
        self._p_gains = self._p_gains * 2  # default p_gains
        self._d_gains = np.zeros(self.ros_node.NUM_JOINTS, dtype=np.float32)  # default d_gains

    def step(self) -> tuple[np.ndarray, bool]:
        """Run a single step of the cold start agent. This will turn on the motors to a desired position.
        The desired position is defined in the robot configuration.
        """
        if self.step_count < self.start_steps:
            actions = self.start_dof_pos + self.dof_pos_err * (self.step_count + 1) / self.start_steps
            done = False
            self.step_count += 1
        else:
            dof_pos_err = self.joint_target_pos - self._get_joint_pos_rel_obs()
            err_large_mask = np.abs(dof_pos_err) > self.dof_max_err
            done = not err_large_mask.any()
            if not done:
                print(
                    f"Current ColdStartAgent gets max error {np.round(np.max(np.abs(dof_pos_err)), decimals=2)}",
                    end="\r",
                )
                raise SystemExit
            else:
                actions = np.zeros(self.ros_node.NUM_JOINTS)

        return actions, done

    def reset(self):
        """Reset the agent."""
        self.start_dof_pos = self._get_joint_pos_rel_obs()
        self.dof_pos_err = self.joint_target_pos - self.start_dof_pos
        self.step_count = 0
