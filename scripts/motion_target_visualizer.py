import os
import signal
import subprocess
import time

import numpy as np
import onnxruntime as ort
import quaternion
import rclpy
import yaml
from geometry_msgs.msg import Point, Quaternion, TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster

from motion_target_msgs.msg import MotionFrame, MotionSequence


def launch_process(command):
    # Start the process
    process = subprocess.Popen(command)

    def handle_signal(signum, frame):
        # Forward the signal to the child process
        process.send_signal(signum)
        # Wait a moment for the process to handle it
        time.sleep(0.1)
        # Exit the parent if the child has exited
        if process.poll() is not None:
            exit(0)

    # Set up signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    return process


class MotionTargetVisualizer(Node):
    """This is a node that publishes the target motion sequence to Robot Joint State for further visualization in RViz."""

    def start_ros_handlers(self):
        """Start ROS handlers."""
        self.joint_state_publisher = self.create_publisher(JointState, "joint_states", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.motion_target_subscriber = self.create_subscription(
            MotionSequence, "motion_target", self.motion_target_callback, 10
        )
        self.main_loop_timer = self.create_timer(0.02, self.main_loop_callback)
        self.get_logger().info("MotionTargetVisualizer started.")

    def motion_target_callback(self, msg: MotionSequence):
        self.motion_target = msg
        self.motion_receive_time = self.get_clock().now()
        self.get_logger().info(f"Received new motion sequence with {len(msg.data)} frames.", once=True)

    def main_loop_callback(self):
        if not hasattr(self, "motion_target"):
            return
        time_from_receive = (self.get_clock().now() - self.motion_receive_time).nanoseconds / 1e9

        frame_selection = None
        min_time_to_go = float("inf")
        times_to_go = []
        for frame in self.motion_target.data:
            time_to_go = frame.time_to_target - time_from_receive
            times_to_go.append(time_to_go)
            if (time_to_go > 0) and (time_to_go < min_time_to_go):
                min_time_to_go = time_to_go
                frame_selection = frame
        if min_time_to_go == float("inf"):
            self.get_logger().info("No valid motion frame found, skipping visualization.", throttle_duration_sec=5.0)
            times_to_go_str = [f"{t:.1e}" for t in times_to_go]
            self.get_logger().debug(f"times_to_go: {times_to_go_str}", throttle_duration_sec=5.0)
            return

        # Publish the joint state
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.header.frame_id = "base_link"
        joint_state.name = self.motion_target.joint_names
        joint_state.position = [frame_selection.joint_pos[i] for i in range(len(self.motion_target.joint_names))]
        joint_state.velocity = [0.0 for i in range(len(self.motion_target.joint_names))]
        joint_state.effort = [0.0 for i in range(len(self.motion_target.joint_names))]
        self.joint_state_publisher.publish(joint_state)

        # Publish the TF transforms based on root rotation
        quat_w = frame_selection.quat_w
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = "world"
        tf_msg.child_frame_id = "torso_link"
        tf_msg.transform.translation.x = 0.0
        tf_msg.transform.translation.y = 0.0
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation = Quaternion(x=quat_w.x, y=quat_w.y, z=quat_w.z, w=quat_w.w)
        self.tf_broadcaster.sendTransform(tf_msg)


def main(args):
    rclpy.init()

    node = MotionTargetVisualizer("motion_target_visualizer")

    node.start_ros_handlers()

    if args.rviz:
        launch_process(
            [
                "ros2",
                "launch",
                "g1_description",
                "rviz.py",
            ]
        )

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Motion Target Visualizer Node")
    parser.add_argument(
        "--rviz",
        action="store_true",
        default=False,
        help="Launch RViz automatically by this process.",
    )

    args = parser.parse_args()

    main(args)
