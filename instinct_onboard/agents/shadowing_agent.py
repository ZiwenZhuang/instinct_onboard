from __future__ import annotations

import os

import numpy as np
import onnxruntime as ort
import rosbag2_py
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
        motion_reference_topic: str = "/motion_reference",
    ):
        super().__init__(logdir, ros_node)
        self.ort_sessions = dict()
        self.motion_reference_topic = motion_reference_topic
        self._parse_obs_config()
        self._load_model()

    def _parse_obs_config(self):
        super()._parse_obs_config()
        with open(os.path.join(self.logdir, "params", "agent.yaml")) as f:
            self.agent_cfg = yaml.unsafe_load(f)
        self.motion_ref_obs_names = self.agent_cfg["policy"]["encoder_cfgs"]["motion_ref"]["component_names"]
        self.ros_node.get_logger().info(f"ShadowingAgent observation names: {self.motion_ref_obs_names}")
        all_obs_names = list(self.obs_funcs.keys())
        self.proprio_obs_names = [obs_name for obs_name in all_obs_names if obs_name not in self.motion_ref_obs_names]
        self.ros_node.get_logger().info(f"ShadowingAgent proprioception observation names: {self.proprio_obs_names}")

    def _load_model(self):
        """Load the ONNX model for the agent."""
        # load ONNX models
        actor_path = os.path.join(self.logdir, "models", "actor.onnx")
        self.ort_sessions["actor"] = ort.InferenceSession(actor_path)
        motion_ref_path = os.path.join(self.logdir, "models", "0-motion_ref.onnx")
        self.ort_sessions["motion_ref"] = ort.InferenceSession(motion_ref_path)
        print(f"Loaded ONNX models from {self.logdir}")
        # load motion sequence as rosbag
        motion_seq_path = os.path.join(
            self.logdir, "rosbag", "model_145000_CMU-140-08_0", "model_145000_CMU-140-08_0.mcap"
        )
        self.bag_reader = rosbag2_py.SequentialReader()
        self.bag_reader.open(
            rosbag2_py.StorageOptions(uri=motion_seq_path, storage_id="mcap"),
            rosbag2_py.ConverterOptions(
                input_serialization_format="cdr",
                output_serialization_format="cdr",
            ),
        )

    def reset(self):
        """Reset the agent state and the rosbag reader."""
        self.reset_rosbag_reader()
        self.motion_reference_buffer = None
        self.pose_w_reference_buffer = None

    def step(self):
        """Perform a single step of the agent."""
        self.advance_rosbag_reader()
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

    def _get_time_to_target_obs(self):
        pass

    def _get_time_from_ref_update_obs(self):
        pass

    def _get_pose_ref_obs(self):
        pass

    def _get_pose_ref_mask_obs(self):
        pass

    def _get_dof_pos_ref_obs(self):
        pass

    def _get_dof_pos_err_ref_obs(self):
        pass

    def _get_dof_pos_mask_obs(self):
        pass

    def _get_link_pos_ref_obs(self):
        pass

    def _get_link_pos_err_ref_obs(self):
        pass

    def _get_link_ref_mask_obs(self):
        pass

    """
    Functions to handle the motion sequence from the rosbag.
    """

    def reset_rosbag_reader(self):
        """Initialize the rosbag reader to read the motion sequence."""
        self.bag_reader.reset()
        self.bag_reader.seek(0)
        assert self.bag_reader.has_next(), "No messages in the rosbag"

        topic_collected_mask = [False, False]  # [motion_reference, pose_w_reference]
        while not all(topic_collected_mask):
            # read next message
            topic, data, timestamp = self.bag_reader.read_next()
            if topic == self.motion_reference_topic:
                self._motion_reference_callback(MotionSequence.deserialize(data))
                topic_collected_mask[0] = True
            elif topic == self.pose_w_reference_topic:
                self._pose_w_reference_callback(PoseArray.deserialize(data))
                topic_collected_mask[1] = True
            else:
                print(f"Skipping topic {topic} in the rosbag")

    # def _motion_reference_callback(self, msg: MotionSequence):
    #     self.motion_reference_buffer = msg

    # def _pose_w_reference_callback(self, msg):
    #     self.pose_w_reference_buffer = msg

    # def advance_rosbag_reader(self):
    #     """Advance the rosbag reader to the next message."""
    #     if self.bag_reader.has_next():
    #         topic, data, timestamp = self.bag_reader.read_next()
    #         if topic == self.motion_reference_topic:
    #             self._motion_reference_callback(MotionSequence.deserialize(data))
    #         # elif topic == self.pose_w_reference_topic:
    #         #     self._pose_w_reference_callback(PoseArray.deserialize(data))
    #         else:
    #             print(f"Skipping topic {topic} in the rosbag")
    #         return True
    #     else:
    #         print("No more messages in the rosbag")
    #         return False
