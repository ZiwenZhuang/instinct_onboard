import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Tuple

import numpy as np
import yaml

from instinct_onboard.utils import CircularBuffer
from instinct_onboard.ros_nodes.ros_real import Ros2Real


class OnboardAgent(ABC):
    """Base class for onboard agents.
    This class is intended to be inherited by specific agents that will handle
    the interaction with the robot and the ONNX model.
    """

    def __init__(self, logdir: str, ros_node: Ros2Real):
        """Initialize the agent with the log directory and ROS node.
        Args:
            logdir (str): The directory where the agent data is stored.
            ros_node (Ros2Real): The ROS node instance to interact with the robot.
        """
        self.logdir = logdir
        self.ros_node = ros_node
        assert isinstance(self.ros_node, Ros2Real), "ros_node must be an instance of Ros2Real"
        env_yaml = os.path.join(self.logdir, "params", "env.yaml")
        with open(env_yaml) as f:
            self.cfg = yaml.unsafe_load(f)

    def _parse_obs_config(self):
        """Parse, set attributes from config dict, initialize buffers to speed up the computation"""
        self.obs_funcs = OrderedDict()
        self.obs_clip = dict()
        self.obs_scales = dict()
        self.obs_history_buffers = dict()
        for obs_name, obs_config in self.cfg["observations"]["policy"].items():
            if (
                obs_name == "concatenate_terms"
                or obs_name == "enable_corruption"
                or obs_name == "history_length"
                or obs_name == "flatten_history_dim"
            ):
                continue
            obs_func: str = obs_config["func"].split(":")[-1]  # get the function name from the config
            # self.obs_funcs will be update in these functions in the order of the config
            if "generated_commands" in obs_func:
                self._parse_generated_commands(obs_name, obs_config)
            else:
                self._parse_observation_function(obs_name, obs_config)
            if obs_config.get("clip", None) is not None:
                self.obs_clip[obs_name] = obs_config["clip"]
            if obs_config.get("scale", None) is not None:
                self.obs_scales[obs_name] = obs_config["scale"]
            if (
                obs_config.get("history_length", 0) != 0
                or self.cfg["observations"]["policy"].get("history_length", None) is not None
            ):
                # if obs_config.get("history_length", None) is not None, use it
                # otherwise, use the global history length
                self.obs_history_buffers[obs_name] = CircularBuffer(
                    obs_config.get("history_length", self.cfg["observations"]["policy"]["history_length"]),
                )

    def _parse_generated_commands(self, obs_name: str, obs_config: dict):
        """Parse the generated commands observation configuration.
        e.g. obs_name: "joint_command", obs_config: joint_command (class -> dict)
             obs_config["func"]:"generated_commands", obs_config["params"]["command_name"]: joint_pos_command
        """
        command_name = obs_config["params"]["command_name"]  # e.g. joint_pos_command
        if hasattr(self, f"_get_{command_name}_obs"):
            self.obs_funcs[obs_name] = getattr(self, f"_get_{command_name}_obs")
        else:
            raise ValueError(
                f"Generated command observation function '{command_name}' not found in the agent. "
                "Please check the configuration."
            )

    def _parse_observation_function(self, obs_name: str, obs_config: dict):
        obs_func = obs_config["func"].split(":")[-1]  # get the function name from the config
        """Parse the observation function from the config."""
        if hasattr(self, f"_get_{obs_func}_obs"):
            self.obs_funcs[obs_name] = getattr(self, f"_get_{obs_func}_obs")
        elif hasattr(self.ros_node, f"_get_{obs_func}_obs"):
            self.obs_funcs[obs_name] = getattr(self.ros_node, f"_get_{obs_func}_obs")
        else:
            raise ValueError(
                f"Observation function '{obs_func}' not found in the agent or ros_node. Please check the configuration."
            )

    def _get_single_obs_term(
        self,
        obs_name: str,
    ) -> np.ndarray:
        """Get a single observation term by its name. It only perform the post-processing operations
        when specified and available.
        """
        obs_value = self.obs_funcs[obs_name]()
        if obs_name in self.obs_clip:
            obs_value = np.clip(obs_value, -self.obs_clip[obs_name], self.obs_clip[obs_name])
        if obs_name in self.obs_scales:
            obs_value *= self.obs_scales[obs_name]
        if obs_name in self.obs_history_buffers:
            # NOTE: this function automatically handles the history buffer
            self.obs_history_buffers[obs_name].append(obs_value)
            obs_value = self.obs_history_buffers[obs_name].buffer
        return obs_value

    def _get_observation(self) -> np.ndarray:
        """Get all observations in the order of the config for the policy.
        Returns:
            np.ndarray: A single vector containing all observations with shape (dim,).
        """
        obs = []
        for obs_name in self.obs_funcs.keys():
            obs_value = self._get_single_obs_term(obs_name)
            obs.append(obs_value.flatten())  # Ensure obs is a 1D vector
        obs = np.concatenate(obs, axis=-1)  # Concatenate all observations into a single vector
        return obs

    @abstractmethod
    def step(self) -> Tuple[np.ndarray, bool]:
        """Run a single step of the ONNX model and return the resulting action and whether the motion is done.
        Let the ROS node handle the action to the motor.
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset the agent. This is a placeholder for any reset logic if needed."""
        pass


class ColdStartAgent(OnboardAgent):
    def __init__(self, startup_step_size: float, ros_node: Ros2Real, joint_target_pos: np.array = None):
        """Initialize the cold start agent.
        Args:
            startup_step_size (float): The step size for the cold start agent to move the joints.
            ros_node (Ros2Real): The ROS node instance to interact with the robot.
        """
        self.ros_node = ros_node
        self.startup_step_size = startup_step_size
        self.joint_target_pos = np.zeros(self.ros_node.NUM_JOINTS) if joint_target_pos is None else joint_target_pos

    def step(self) -> Tuple[np.ndarray, bool]:
        """Run a single step of the cold start agent. This will turn on the motors to a desired position.
        The desired position is defined in the robot configuration.
        """
        dof_pos_err = self.joint_target_pos - self.ros_node._get_joint_pos_rel_obs()
        err_large_mask = np.abs(dof_pos_err) > self.startup_step_size
        done = not err_large_mask.any()
        if not done:
            print(
                f"Current ColdStartAgent gets max error {np.round(np.max(np.abs(dof_pos_err)), decimals=2)}", end="\r"
            )
        dof_pos_target = np.where(
            err_large_mask,
            self.ros_node._get_joint_pos_rel_obs() + np.sign(dof_pos_err) * self.startup_step_size,
            self.joint_target_pos,
        )
        actions = (dof_pos_target) / self.ros_node.action_scale

        return actions, done

    def reset(self):
        """Reset the agent. This is a placeholder for any reset logic if needed.
        Currently, no reset logic is implemented.
        """
        pass
