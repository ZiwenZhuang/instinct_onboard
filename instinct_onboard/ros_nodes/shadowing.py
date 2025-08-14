import numpy as np
import quaternion
import rclpy
from rclpy.node import Node

from instinct_onboard.utils import quat_to_tan_norm_batch
from motion_target_msgs.msg import MotionSequence

from .ros_real import Ros2Real


class ShadowingNodeMixin:
    def __init__(self, *args, motion_sequence_topic: str = "/motion_target", **kwargs):
        """Initialize the ShadowingNodeMixin with the motion sequence topic."""
        super().__init__(*args, **kwargs)
        self.motion_sequence_topic = motion_sequence_topic
        self.motion_sequence_buffer = None

    def start_ros_handlers(self):
        """Start the ROS handlers for shadowing and call the super class method."""

        self.motion_sequence_subscriber = self.create_subscription(
            MotionSequence, self.motion_sequence_topic, self._motion_sequence_callback, 10
        )

        super().start_ros_handlers()

    def _motion_sequence_callback(self, msg: MotionSequence):
        """Callback for the motion sequence topic."""
        # This method is called when a new motion sequence is received.
        # It should handle the message and update the internal state accordingly.
        self.motion_sequence_buffer = msg
        self.motion_sequence_receive_time = self.get_clock().now()

        # pack motion_sequence into a dict of numpy arrays
        num_frames = len(msg.data)
        num_joints = len(msg.data[0].joint_pos)
        num_links = len(msg.data[0].link_pos)
        time_to_target = np.zeros(num_frames, dtype=np.float32)
        root_pos_b = np.zeros((num_frames, 3), dtype=np.float32)
        root_quat_w = np.zeros((num_frames, 4), dtype=np.float32)
        pose_mask = np.ones((num_frames, 4), dtype=np.float32)  # (num_frames, 4), for root_pos_b and root_quat_b
        joint_pos = np.zeros((num_frames, num_joints), dtype=np.float32)
        joint_pos_mask = np.ones((num_frames, num_joints), dtype=np.float32)  # (num_frames, num_joints)
        link_pos = np.zeros((num_frames, num_links, 3), dtype=np.float32)
        link_pos_mask = np.ones((num_frames, num_links), dtype=np.float32)  # (num_frames, num_links)
        link_quat = np.zeros((num_frames, num_links, 4), dtype=np.float32)
        link_tannorm = np.zeros((num_frames, num_links, 6), dtype=np.float32)
        link_quat_mask = np.ones((num_frames, num_links), dtype=np.float32)  # (num_frames, num_links, 4)
        for i, frame in enumerate(msg.data):
            time_to_target[i] = frame.time_to_target
            root_pos_b[i, 0] = frame.pos_b.x
            root_pos_b[i, 1] = frame.pos_b.y
            root_pos_b[i, 2] = frame.pos_b.z
            root_quat_w[i, 0] = frame.quat_w.w
            root_quat_w[i, 1] = frame.quat_w.x
            root_quat_w[i, 2] = frame.quat_w.y
            root_quat_w[i, 3] = frame.quat_w.z
            for j in range(num_joints):
                joint_pos[i, j] = frame.joint_pos[j]  # in urdf space, simulation order.
                joint_pos_mask[i, j] = frame.joint_pos_mask[j]  # (num_frames, num_joints)
            for j in range(num_links):
                link_pos[i, j, 0] = frame.link_pos[j].x
                link_pos[i, j, 1] = frame.link_pos[j].y
                link_pos[i, j, 2] = frame.link_pos[j].z
                link_quat[i, j, 0] = frame.link_quat[j].w
                link_quat[i, j, 1] = frame.link_quat[j].x
                link_quat[i, j, 2] = frame.link_quat[j].y
                link_quat[i, j, 3] = frame.link_quat[j].z
                link_pos_mask[i, j] = frame.link_pos_mask[j]
                link_quat_mask[i, j] = frame.link_quat_mask[j]
            pose_mask[i] = frame.pose_mask  # (4,) for root_pos_b and root_quat_w
            link_tannorm[i] = quat_to_tan_norm_batch(link_quat[i])

        self.packed_motion_sequence_buffer = {
            "time_to_target_when_received": time_to_target,
            "time_to_target": time_to_target,
            "root_pos_b": root_pos_b,
            "root_quat_w": root_quat_w,
            "pose_mask": pose_mask,
            "joint_pos": joint_pos,
            "joint_pos_mask": joint_pos_mask,
            "link_pos": link_pos,
            "link_pos_mask": link_pos_mask,
            "link_quat": link_quat,
            "link_quat_mask": link_quat_mask,
            "link_tannorm": link_tannorm,
        }

    def refresh_time_to_target(self):
        """Refresh the 'time_to_target' term in the packed motion sequence buffer."""
        if not hasattr(self, "packed_motion_sequence_buffer"):
            return
        self.packed_motion_sequence_buffer["time_to_target"] = (
            self.packed_motion_sequence_buffer["time_to_target_when_received"]
            - (self.get_clock().now() - self.motion_sequence_receive_time).nanoseconds / 1e9
        )


class ShadowingNode(ShadowingNodeMixin, Ros2Real):
    pass
