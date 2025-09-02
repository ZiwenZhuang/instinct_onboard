import numpy as np
from std_msgs.msg import Float32MultiArray

from .ros_real import Ros2Real


class ParkourNodeMixin:
    def __init__(self, *args, depth_latent_topic: str = "/depth_latent", **kwargs):
        """Initialize the ParkourNodeMixin with the depth latent topic."""
        super().__init__(*args, **kwargs)
        self.joy_stick_command = [0, 0, 0, 0]
        self.depth_latent_topic = depth_latent_topic
        self.depth_latent_buffer = None

    def start_ros_handlers(self):
        """Start the ROS handlers for parkour and call the super class method."""

        self.depth_latent_subscriber = self.create_subscription(
            Float32MultiArray, self.depth_latent_topic, self._depth_latent_callback, 10
        )

        super().start_ros_handlers()

    def _joy_stick_callback(self, msg):
        super()._joy_stick_callback(msg)
        self.joy_stick_command = [msg.lx, msg.ly, msg.rx, msg.ry]

    def _depth_latent_callback(self, msg: Float32MultiArray):
        """Callback for the depth latent topic."""
        # This method is called when a new depth latent message is received.
        # It should handle the message and update the internal state accordingly.
        self.depth_latent_buffer = np.array(msg.data, dtype=np.float32)
        self.depth_latent_receive_time = self.get_clock().now()


class ParkourNode(ParkourNodeMixin, Ros2Real):
    pass
