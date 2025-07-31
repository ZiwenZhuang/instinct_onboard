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


class ShadowingNode(ShadowingNodeMixin, Ros2Real):
    pass
