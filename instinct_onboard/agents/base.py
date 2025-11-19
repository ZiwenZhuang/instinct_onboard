import os
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Callable, Tuple

import numpy as np
import yaml

from instinct_onboard.ros_nodes.base import RealNode
from instinct_onboard.utils import CircularBuffer


class OnboardAgent(ABC):
    """Base class for onboard agents.
    This class is intended to be inherited by specific agents that will handle
    the interaction with the robot and the ONNX model.
    """

    def __init__(self, logdir: str, ros_node: RealNode):
        """Initialize the agent with the log directory and ROS node.
        Args:
            logdir (str): The directory where the agent data is stored.
            ros_node (Ros2Real): The ROS node instance to interact with the robot.
        """
        self.logdir = logdir
        self.ros_node: RealNode = ros_node
        assert isinstance(self.ros_node, RealNode), "ros_node must be an instance of RealNode"
        env_yaml = os.path.join(self.logdir, "params", "env.yaml")
        with open(env_yaml) as f:
            self.cfg = yaml.unsafe_load(f)

    def _parse_action_config(self):
        """Parse control-related configurations from the environment YAML file."""

        # default joint positions
        self.default_joint_pos = np.zeros(self.ros_node.NUM_JOINTS, dtype=np.float32)
        for joint_name_expr, joint_pos in self.cfg["scene"]["robot"]["init_state"]["joint_pos"].items():
            # compute the default joint pos from configuration for articulation.default_joint_pos
            for i in range(self.ros_node.NUM_JOINTS):
                name = self.ros_node.sim_joint_names[i]
                if re.search(joint_name_expr, name):
                    self.default_joint_pos[i] = joint_pos

        # default joint velocities
        self.default_joint_vel = np.zeros(self.ros_node.NUM_JOINTS, dtype=np.float32)
        for joint_name_expr, joint_vel in self.cfg["scene"]["robot"]["init_state"]["joint_vel"].items():
            for i in range(self.ros_node.NUM_JOINTS):
                name = self.ros_node.sim_joint_names[i]
                if re.search(joint_name_expr, name):
                    self.default_joint_vel[i] = joint_vel

        # stiffness and damping gains
        self._p_gains = np.zeros(self.ros_node.NUM_JOINTS, dtype=np.float32)
        self._d_gains = np.zeros(self.ros_node.NUM_JOINTS, dtype=np.float32)
        for actuator_name, actuator_config in self.cfg["scene"]["robot"]["actuators"].items():
            print(f"Get {actuator_config['class_type']} in actuator {actuator_name}")
            for i in range(self.ros_node.NUM_JOINTS):
                name = self.ros_node.sim_joint_names[i]
                for _, joint_name_expr in enumerate(actuator_config["joint_names_expr"]):
                    if re.search(joint_name_expr, name):
                        if isinstance(actuator_config["stiffness"], dict):
                            for key, value in actuator_config["stiffness"].items():
                                if re.search(key, name):
                                    self._p_gains[i] = value
                        else:
                            self._p_gains[i] = actuator_config["stiffness"]
                        if isinstance(actuator_config["damping"], dict):
                            for key, value in actuator_config["damping"].items():
                                if re.search(key, name):
                                    self._d_gains[i] = value
                        else:
                            self._d_gains[i] = actuator_config["damping"]

        # action scale
        self._action_scale = np.zeros(self.ros_node.NUM_ACTIONS, dtype=np.float32)
        self._action_offset = self.default_joint_pos.copy()  # default action offset in robot urdf coordinate
        for action_names, action_config in self.cfg["actions"].items():
            if not action_config["asset_name"] == "robot":
                continue
            for i in range(self.ros_node.NUM_JOINTS):
                name = self.ros_node.sim_joint_names[i]
                for _, joint_name_expr in enumerate(action_config["joint_names"]):
                    if re.search(joint_name_expr, name):
                        if isinstance(action_config["scale"], dict):
                            for key, value in action_config["scale"].items():
                                if re.search(key, name):
                                    self._action_scale[i] = value
                        else:
                            self._action_scale[i] = action_config["scale"]
                        # print("Joint {}({}) has action scale {}".format(i, name, self.action_scale[i]))
                    if not action_config["use_default_offset"]:
                        # not using articulation.default_joint_pos as default offset
                        if isinstance(action_config["offset"], dict):
                            self._action_offset[i] = action_config["offset"][joint_name_expr]
                        else:
                            self._action_offset[i] = action_config["offset"]

    def _parse_obs_config(self):
        """Parse, set attributes from config dict, initialize buffers to speed up the computation"""
        self.obs_funcs: OrderedDict[str, Callable] = OrderedDict()
        self.obs_clip: dict[str, float] = dict()
        self.obs_scales: dict[str, float] = dict()
        self.obs_history_buffers: dict[str, CircularBuffer] = dict()
        for obs_name, obs_config in self.cfg["observations"]["policy"].items():
            if (
                obs_name == "concatenate_terms"
                or obs_name == "enable_corruption"
                or obs_name == "history_length"
                or obs_name == "flatten_history_dim"
                or obs_config is None
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
        if hasattr(self, f"_get_{command_name}_cmd_obs"):
            self.obs_funcs[obs_name] = getattr(self, f"_get_{command_name}_cmd_obs")
        elif hasattr(self, f"_get_{command_name}_obs"):
            self.obs_funcs[obs_name] = getattr(self, f"_get_{command_name}_obs")
            print(
                "Warning: '_get_{command_name}_obs' for command {command_name} shall be deprecated. Please implementing your command-related observation function '_get_{command_name}_cmd_obs' instead."
            )
        else:
            raise ValueError(
                f"Generated command observation function '_get_{command_name}_obs' not found in the agent. "
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
                f"Observation function '_get_{obs_func}_obs' not found in the agent or ros_node. Please check the"
                " configuration."
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

    def _build_obs_shapes(self) -> None:
        """Build the obs_shapes if not exists. Please make sure this functions is called before self.reset() procedures."""
        if not hasattr(self, "obs_shapes"):
            self.obs_shapes: OrderedDict[str, tuple] = OrderedDict()
            for obs_name in self.obs_funcs.keys():
                self.obs_shapes[obs_name] = self._get_single_obs_term(obs_name).shape  # (dim,)

    def _get_obs_slice(self, obs_name: str) -> slice:
        """Get the slice of the observation term by its name in the concatenated observation vector."""
        assert hasattr(self, "obs_shapes"), "obs_shapes must be built before calling this function"
        obs_term_names = list(self.obs_funcs.keys())
        target_obs_term_idx = obs_term_names.index(obs_name)
        start_idx = sum(np.prod(self.obs_shapes[obs_name]) for obs_name in obs_term_names[:target_obs_term_idx])
        end_idx = start_idx + np.prod(self.obs_shapes[obs_name])
        return slice(start_idx, end_idx)

    @abstractmethod
    def step(self) -> Tuple[np.ndarray, bool]:
        """Run a single step of the ONNX model and return the resulting action and whether the motion is done.
        Let the ROS node handle the action to the motor.
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset the agent. This is a placeholder for any reset logic if needed."""
        self._build_obs_shapes()
        for obs_history_buffer in self.obs_history_buffers.values():
            obs_history_buffer.reset()

    @property
    def action_scale(self) -> np.ndarray:
        return self._action_scale

    @property
    def action_offset(self) -> np.ndarray:
        return self._action_offset

    @property
    def p_gains(self) -> np.ndarray:
        """Get the proportional gains for the PD controller."""
        return self._p_gains

    @property
    def d_gains(self) -> np.ndarray:
        """Get the derivative gains for the PD controller."""
        return self._d_gains

    """
    Some observation functions that depends on the agent's cfg
    """

    def _get_joint_pos_rel_obs(self) -> np.ndarray:
        """Get the joint position relative to the default_joint_pos."""
        return self.ros_node.joint_pos_ - self.default_joint_pos  # shape (NUM_JOINTS,)

    def _get_joint_vel_rel_obs(self) -> np.ndarray:
        """Get the joint velocity relative to the default_joint_pos."""
        return self.ros_node.joint_vel_ - self.default_joint_vel  # shape (NUM_JOINTS,)


class ColdStartAgent(OnboardAgent):
    def __init__(
        self,
        startup_step_size: float,
        ros_node: RealNode,
        joint_target_pos: np.array = None,
        action_scale: np.array = None,
        action_offset: np.array = None,
        p_gains: np.array = None,
        d_gains: np.array = None,
    ):
        """Initialize the cold start agent.
        Args:
            startup_step_size (float): The step size for the cold start agent to move the joints.
            ros_node (RealNode): The ROS node instance to interact with the robot.
            action_scale:
            action_offset: Due to the last_action sharing mechanism, it is sometimes needed to specify the
                action scale and action offset in the cold-start agent.
        """
        self.ros_node = ros_node
        self.startup_step_size = startup_step_size
        self.joint_target_pos = np.zeros(self.ros_node.NUM_JOINTS) if joint_target_pos is None else joint_target_pos
        self._action_scale = (
            np.zeros(self.ros_node.NUM_JOINTS, dtype=np.float32) if action_scale is None else action_scale
        )
        self._action_offset = (
            np.ones(self.ros_node.NUM_JOINTS, dtype=np.float32) if action_offset is None else action_offset
        )
        self._p_gains = np.ones(self.ros_node.NUM_JOINTS, dtype=np.float32) * 10.0 if p_gains is None else p_gains
        self._d_gains = np.zeros(self.ros_node.NUM_JOINTS, dtype=np.float32) if d_gains is None else d_gains

    def step(self) -> Tuple[np.ndarray, bool]:
        """Run a single step of the cold start agent. This will turn on the motors to a desired position.
        The desired position is defined in the robot configuration.
        """
        dof_pos_err = self.joint_target_pos - self.ros_node._get_joint_pos_obs()
        err_large_mask = np.abs(dof_pos_err) > self.startup_step_size
        done = not err_large_mask.any()
        if not done:
            max_err_idx = np.argmax(np.abs(dof_pos_err))
            print(
                f"Current ColdStartAgent gets max error {np.round(np.max(np.abs(dof_pos_err)), decimals=3):.3f} at sim joint {max_err_idx:2d}, should be {self.joint_target_pos[max_err_idx]:.3f} but currently is {self.ros_node._get_joint_pos_obs()[max_err_idx]:.3f}",
                end="\r",
            )
        dof_pos_target = np.where(
            err_large_mask,
            self.ros_node._get_joint_pos_obs() + np.sign(dof_pos_err) * self.startup_step_size,
            self.joint_target_pos,
        )
        actions = (dof_pos_target - self._action_offset) / self._action_scale

        return actions, done

    def reset(self):
        """Reset the agent. This is a placeholder for any reset logic if needed.
        Currently, no reset logic is implemented.
        """
        pass
