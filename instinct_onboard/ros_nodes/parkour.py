import numpy as np
from std_msgs.msg import Float32MultiArray

import instinct_onboard.robot_cfgs as robot_cfgs

from .ros_real import Ros2Real


class ParkourNodeMixin:
    def __init__(self, *args, depth_latent_topic: str = "/depth_latent", **kwargs):
        """Initialize the ParkourNodeMixin with the depth latent topic."""
        super().__init__(*args, **kwargs)
        self.joy_stick_command = [0, 0, 0, 0]  # [lx, ly, rx, ry]
        self.joy_stick_counter = [0, 0, 0, 0]  # [up, down, left, right]
        self.joy_stick_button_flag = [False, False, False, False]  # [up, down, left, right]
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
        buttons = robot_cfgs.WirelessButtons

        if msg.keys & buttons.up:
            if not self.joy_stick_button_flag[0]:
                self.joy_stick_counter[0] += 1
                self.joy_stick_button_flag[0] = True
        else:
            self.joy_stick_button_flag[0] = False

        if msg.keys & buttons.down:
            if not self.joy_stick_button_flag[1]:
                self.joy_stick_counter[1] += 1
                self.joy_stick_button_flag[1] = True
        else:
            self.joy_stick_button_flag[1] = False

        if msg.keys & buttons.left:
            if not self.joy_stick_button_flag[2]:
                self.joy_stick_counter[2] += 1
                self.joy_stick_button_flag[2] = True
        else:
            self.joy_stick_button_flag[2] = False

        if msg.keys & buttons.right:
            if not self.joy_stick_button_flag[3]:
                self.joy_stick_counter[3] += 1
                self.joy_stick_button_flag[3] = True
        else:
            self.joy_stick_button_flag[3] = False

    def _depth_latent_callback(self, msg: Float32MultiArray):
        """Callback for the depth latent topic."""
        # This method is called when a new depth latent message is received.
        # It should handle the message and update the internal state accordingly.
        self.depth_latent_buffer = np.array(msg.data, dtype=np.float32)
        self.depth_latent_receive_time = self.get_clock().now()


class ParkourNode(ParkourNodeMixin, Ros2Real):
    pass
