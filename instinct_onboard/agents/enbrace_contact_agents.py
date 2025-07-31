from __future__ import annotations

import os

import numpy as np
import onnxruntime as ort
import rosbag2_py
from geometry_msgs.msg import PoseArray

from instinct_onboard.agents.base import OnboardAgent
from instinct_onboard.ros_nodes.ros_real import Ros2Real
from motion_target_msgs.msg import MotionSequence


class EmbraceContactAgent(OnboardAgent):
    def __init__(
        self,
        logdir: str,
        ros_node: Ros2Real,
        motion_reference_topic: str = "/motion_reference",
        pose_w_reference_topic: str = "/global_rotation_reference",
    ):
        super().__init__(logdir, ros_node)
        self.ort_sessions = dict()
        self.motion_reference_topic = motion_reference_topic
        self.pose_w_reference_topic = pose_w_reference_topic
        self._parse_obs_config()
        self._load_model()

    def _load_model(self):
        """Load the ONNX model for the agent."""
        # load ONNX models
        actor_path = os.path.join(self.logdir, "models", "actor.onnx")
        self.ort_sessions["actor"] = ort.InferenceSession(actor_path)
        motion_ref_path = os.path.join(self.logdir, "models", "motion_ref.onnx")
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

    """
    Agent specific observation functions for Embrace Contact Agent.
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

    def _motion_reference_callback(self, msg: MotionSequence):
        self.motion_reference_buffer = msg

    def _pose_w_reference_callback(self, msg):
        self.pose_w_reference_buffer = msg

    def advance_rosbag_reader(self):
        """Advance the rosbag reader to the next message."""
        if self.bag_reader.has_next():
            topic, data, timestamp = self.bag_reader.read_next()
            if topic == self.motion_reference_topic:
                self._motion_reference_callback(MotionSequence.deserialize(data))
            elif topic == self.pose_w_reference_topic:
                self._pose_w_reference_callback(PoseArray.deserialize(data))
            else:
                print(f"Skipping topic {topic} in the rosbag")
            return True
        else:
            print("No more messages in the rosbag")
            return False
