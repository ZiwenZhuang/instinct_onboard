from __future__ import annotations

import os

import numpy as np
import onnxruntime as ort
import quaternion
import yaml
from geometry_msgs.msg import PoseArray

from instinct_onboard.agents.base import OnboardAgent
from instinct_onboard.ros_nodes.ros_real import Ros2Real
from motion_target_msgs.msg import MotionSequence


class ShadowingAgent(OnboardAgent):
    def __init__(
        self,
        logdir: str,
        ros_node: Ros2Real,
    ):
        super().__init__(logdir, ros_node)
        self.ort_sessions = dict()
        self._parse_obs_config()
        self._load_models()

    def _parse_obs_config(self):
        super()._parse_obs_config()
        with open(os.path.join(self.logdir, "params", "agent.yaml")) as f:
            self.agent_cfg = yaml.unsafe_load(f)
        self.motion_ref_obs_names = self.agent_cfg["policy"]["encoder_cfgs"]["motion_ref"]["component_names"]
        self.ros_node.get_logger().info(f"ShadowingAgent observation names: {self.motion_ref_obs_names}")
        all_obs_names = list(self.obs_funcs.keys())
        self.proprio_obs_names = [obs_name for obs_name in all_obs_names if obs_name not in self.motion_ref_obs_names]
        self.ros_node.get_logger().info(f"ShadowingAgent proprioception observation names: {self.proprio_obs_names}")

    def _load_models(self):
        """Load the ONNX model for the agent."""
        # load ONNX models
        ort_execution_providers = ort.get_available_providers()
        actor_path = os.path.join(self.logdir, "exported", "actor.onnx")
        self.ort_sessions["actor"] = ort.InferenceSession(actor_path, providers=ort_execution_providers)
        motion_ref_path = os.path.join(self.logdir, "exported", "0-motion_ref.onnx")
        self.ort_sessions["motion_ref"] = ort.InferenceSession(motion_ref_path, providers=ort_execution_providers)
        fk_path = os.path.join(self.logdir, "exported", "forward_kinematics.onnx")
        self.ort_sessions["fk"] = ort.InferenceSession(fk_path, providers=ort_execution_providers)
        print(f"Loaded ONNX models from {self.logdir}")

    def _update_links_poses(self):
        """Update the current link positions based on self.ros_node.joint_pos_."""
        # get the current joint positions
        joint_pos = self.ros_node.joint_pos_
        # run forward kinematics to get the link positions
        fk_input_name = self.ort_sessions["fk"].get_inputs()[0].name
        output = self.ort_sessions["fk"].run(None, {fk_input_name: joint_pos})
        link_pos, link_quat = output[0], output[1]  # link_pos: (num_links, 3), link_quat: (num_links, 4)
        self.link_pos_ = link_pos
        self.link_quat_ = link_quat

    def reset(self):
        """Reset the agent state and the rosbag reader."""
        pass

    def step(self):
        """Perform a single step of the agent."""
        self._update_links_poses()
        # due to the model which reads the motion sequence, and then concat at the end of the proioception vector, we get obs term one by one.

        # pack all motion sequence obs term
        motion_ref_obs = []
        for motion_ref_obs_name in self.motion_ref_obs_names:
            obs_term_value = self._get_single_obs_term(motion_ref_obs_name)
            time_dim = obs_term_value.shape[0]  # (time, batch_size, ...)
            motion_ref_obs.append(obs_term_value.reshape(1, time_dim, -1))  # reshape to (batch_size, time, -1)
        motion_ref_obs = np.concatenate(
            motion_ref_obs, axis=1
        )  # across time dimension. shape (batch_size, time, num_obs_terms)

        # run motion reference encoder
        motion_ref_input_name = self.ort_sessions["motion_ref"].get_inputs()[0].name
        motion_ref_output = self.ort_sessions["motion_ref"].run(None, {motion_ref_input_name: motion_ref_obs})[0]

        # pack actor MLP input
        proprio_obs = []
        for proprio_obs_name in self.proprio_obs_names:
            obs_term_value = self._get_single_obs_term(proprio_obs_name)
            proprio_obs.append(obs_term_value.reshape(1, -1))
        proprio_obs.append(motion_ref_output.reshape(1, -1))  # append motion reference output
        proprio_obs = np.concatenate(proprio_obs, axis=1)

        # run actor MLP
        actor_input_name = self.ort_sessions["actor"].get_inputs()[0].name
        action = self.ort_sessions["actor"].run(None, {actor_input_name: proprio_obs})[0]
        action = action.reshape(-1)
        done = False

        return action, done

    """
    Agent specific observation functions for Shadowing Agent.
    """

    def _get_time_to_target_obs(self) -> np.ndarray:
        """Return shape: (1, num_frames)"""
        return self.ros_node.packed_motion_sequence_buffer["time_to_target"].reshape(-1, 1)  # (num_frames, 1)

    def _get_time_from_ref_update_obs(self):
        return np.array(
            [
                (self.ros_node.get_clock().now().nanoseconds - self.ros_node.motion_sequence_receive_time.nanoseconds)
                / 1e9
            ]
            * self.ros_node.packed_motion_sequence_buffer["time_to_target"].shape[0],
            dtype=np.float32,
        )  # (num_frames,)

    def _get_pose_ref_mask_obs(self):
        return np.ones(
            (self.ros_node.packed_motion_sequence_buffer["time_to_target"].shape[0], 4), dtype=np.float32
        )  # (num_frames, 4)

    def _get_joint_pos_ref_obs(self):
        """Command, return shape: (num_frames, num_joints)"""
        return (
            self.ros_node.packed_motion_sequence_buffer["joint_pos"] - self.ros_node.default_joint_pos[None, :]
        )  # (num_frames, num_joints)

    def _get_joint_pos_err_ref_obs(self):
        """Command, return shape: (num_frames, num_joints)"""
        return (
            self.ros_node.packed_motion_sequence_buffer["joint_pos"] - self.ros_node.joint_pos_[None, :]
        )  # (num_frames, num_joints)

    def _get_joint_pos_mask_obs(self):
        """Command, return shape: (num_frames, num_joints)"""
        return np.ones_like(
            self.ros_node.packed_motion_sequence_buffer["joint_pos"],
            dtype=np.float32,
        )

    def _get_link_pos_ref_obs(self):
        return self.ros_node.packed_motion_sequence_buffer["link_pos"]  # (num_frames, num_links, 3), in robot base link

    def _get_link_pos_err_ref_obs(self):
        return (
            self.ros_node.packed_motion_sequence_buffer["link_pos"] - self.link_pos_[None, :, :]
        )  # (num_frames, num_links, 3)

    def _get_link_ref_mask_obs(self):
        return np.ones(
            self.ros_node.packed_motion_sequence_buffer["link_pos"].shape[:2],
            dtype=np.float32,
        )  # (num_frames, num_links)

    def _get_link_rot_ref_obs(self):
        return self.ros_node.packed_motion_sequence_buffer[
            "link_tannorm"
        ]  # (num_frames, num_links, 6), in robot base link

    def _get_link_rot_err_ref_obs(self):
        link_quat_ref = self.ros_node.packed_motion_sequence_buffer["link_quat"]
        link_quat_ = self.link_quat_[None, :, :]  # (1, num_links, 4)
        link_rot_err = quaternion.quaternion_multiply(quaternion.quaternion_conjugate(link_quat_), link_quat_ref)
        link_tannorm_err = np.concatenate(
            [
                quaternion.rotate_vectors(
                    link_rot_err,
                    self.ros_node.tannorm_prototype[0][None, :].repeat(link_quat_ref.shape[1], axis=0)[None, :, :],
                ),  # (num_frames, num_links, 3)
                quaternion.rotate_vectors(
                    link_rot_err,
                    self.ros_node.tannorm_prototype[1][None, :].repeat(link_quat_ref.shape[1], axis=0)[None, :, :],
                ),  # (num_frames, num_links, 3)
            ],
            axis=-1,
        ).astype(np.float32)
        return link_tannorm_err  # (num_frames, num_links, 6)

    def _get_link_rot_mask_obs(self):
        return np.ones(
            self.ros_node.packed_motion_sequence_buffer["link_quat"].shape[:2],
            dtype=np.float32,
        )  # (num_frames, num_links)
