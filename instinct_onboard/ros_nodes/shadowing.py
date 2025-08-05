import numpy as np
import quaternion
import rclpy
from rclpy.node import Node

from motion_target_msgs.msg import MotionSequence

from .ros_real import Ros2Real


class ShadowingNodeMixin:
    def __init__(self, *args, motion_sequence_topic: str = "/motion_sequence", **kwargs):
        """Initialize the ShadowingNodeMixin with the motion sequence topic."""
        super().__init__(*args, **kwargs)
        self.motion_sequence_topic = motion_sequence_topic
        self.motion_sequence_buffer = None
        self.tannorm_prototype = np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32)  # x,y,z,x,y,z, unit length, (2, 3)

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
        root_quat_b = np.zeros((num_frames, 4), dtype=np.float32)
        joint_pos = np.zeros((num_frames, num_joints), dtype=np.float32)
        link_pos = np.zeros((num_frames, num_links, 3), dtype=np.float32)
        link_quat = np.zeros((num_frames, num_links, 4), dtype=np.float32)
        for i, frame in enumerate(msg.data):
            time_to_target[i] = frame.time_to_target
            root_pos_b[i] = frame.root_pos_b
            root_quat_b[i, 0] = frame.root_quat_b.x
            root_quat_b[i, 1] = frame.root_quat_b.y
            root_quat_b[i, 2] = frame.root_quat_b.z
            root_quat_b[i, 3] = frame.root_quat_b.w
            for j in range(num_joints):
                joint_pos[i, j] = frame.joint_pos[j]  # in urdf space, simulation order.
            for j in range(num_links):
                link_pos[i, j] = frame.link_pos[j]
                link_quat[i, j, 0] = frame.link_quat[j].w
                link_quat[i, j, 1] = frame.link_quat[j].x
                link_quat[i, j, 2] = frame.link_quat[j].y
                link_quat[i, j, 3] = frame.link_quat[j].z
        link_tannorm = np.concatenate(
            [
                quaternion.rotate_vectors(
                    link_quat, self.tannorm_prototype[0][None, :].repeat(num_links, axis=0)[None, :, :]
                ),  # (num_frames, num_links, 3)
                quaternion.rotate_vectors(
                    link_quat, self.tannorm_prototype[1][None, :].repeat(num_links, axis=0)[None, :, :]
                ),  # (num_frames, num_links, 3)
            ],
            axis=-1,
        ).astype(
            np.float32
        )  # (num_frames, num_links, 6)

        self.packed_motion_sequence_buffer = {
            "time_to_target_when_received": time_to_target,
            "time_to_target": time_to_target,
            "root_pos_b": root_pos_b,
            "root_quat_b": root_quat_b,
            "joint_pos": joint_pos,
            "link_pos": link_pos,
            "link_quat": link_quat,
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
