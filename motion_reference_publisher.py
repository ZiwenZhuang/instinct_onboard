import os, sys
import asyncio

import rclpy
from rclpy.node import Node
import rclpy.time
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState

from motion_reference_msgs.msg import MotionReference, MotionFrame

import numpy as np
import quaternion
import robot_cfgs

class MotionReferencePublisher(Node):
    """ Display the motion reference data for rviz visualization.
    """
    def __init__(self, args):
        super().__init__("motion_reference_display")
        self.args = args
        self.robot_cfg = getattr(robot_cfgs, args.robot_class)

        self.bag_reader = rosbag2_py.SequentialReader()
        self.bag_reader.open(
            rosbag2_py.StorageOptions(uri=args.bagdir, storage_id="mcap"),
            rosbag2_py.ConverterOptions(
                input_serialization_format="cdr",
                output_serialization_format="cdr",
            ),
        )

        self.motion_reference_pub = self.create_publisher(
            MotionReference,
            "/motion_reference",
            1,
        )

        assert self.publish_a_frame(), "Failed to publish the first frame"
        self.prev_rosbag_msg_time = self.rosbag_msg_time

        if args.debug_vis:
            self.joint_state_msg = JointState()
            self.joint_state_pub = self.create_publisher(
                JointState,
                "joint_states",
                1,
            )
            self.tf_broadcaster = TransformBroadcaster(self)

            # publish the debug vis once
            self.show_robot_reference_callback()
            # start the debug timeer
            self.debug_vis_timer = self.create_timer(
                0.01,
                self.show_robot_reference_callback,
            )

    async def motion_reference_publish(self):
        self.get_logger().info("First frame published, publisher system launched, wait for hitting Enter to continue publishing...")
        user_cmd = input()
        if user_cmd == "q":
            self.get_logger().info("User Cancelled.")
            rclpy.shutdown()
            return

        while self.publish_a_frame() and rclpy.ok():
            duration = self.rosbag_msg_time - self.prev_rosbag_msg_time
            sleep_time_f = (duration).nanoseconds / 1e9
            await asyncio.sleep(sleep_time_f)
            self.prev_rosbag_msg_time = self.rosbag_msg_time
        
        self.get_logger().info("Finished publishing all frames")
        rclpy.shutdown()

    def publish_a_frame(self) -> bool:
        msg = self.get_next_message()
        if msg is not None:
            topic, msg, timestamp = msg
            # Get the rosbag time using timestamp or header.stamp
            if False:
                timestamp_sec = int(timestamp / 1e9); timestamp_nsec = int(timestamp % 1e9)
                self.rosbag_msg_time = rclpy.time.Time(seconds=timestamp_sec, nanoseconds=timestamp_nsec)
            else:
                self.rosbag_msg_time = rclpy.time.Time.from_msg(msg.header.stamp)
            # update the real pulish time.
            self.msg_publish_time = self.get_clock().now()

            # update header timestamp
            msg.header.stamp = self.msg_publish_time.to_msg()
            self.motion_reference_msg = msg
            self.motion_reference_pub.publish(msg)
            self.get_logger().info("Published motion reference message at {:.3f}s".format(self.msg_publish_time.nanoseconds / 1e9))
            return True
        else:
            self.get_logger().info("No more messages to publish")
            delattr(self, "bag_reader")
            return False

    def get_next_message(self):
        if self.bag_reader.has_next():
            topic, data, timestamp = self.bag_reader.read_next()
            if not topic == self.args.topic:
                self.get_logger().warn("Unexpected topic: %s", topic)
                return None
            try:
                msg = deserialize_message(data, MotionReference)
                return topic, msg, timestamp
            except Exception as e:
                self.get_logger().error("Failed to deserialize message: %s", e)
                return None
        return None
        
    def show_robot_reference_callback(self):
        """ update as a timer. """
        # if self.msg_publish_time is None:
        #     self.get_logger().info("No motion reference message published yet")
        #     return

        current_time = self.get_clock().now()
        time_passed_from_motion_reference = current_time - self.msg_publish_time

        # compute which frame should be displayed
        frame_idx = -1
        closest_motion_frame_time = None
        for idx, motion_frame in enumerate(self.motion_reference_msg.data):
            if motion_frame.time_to_target > (time_passed_from_motion_reference.nanoseconds / 1e9):
                if closest_motion_frame_time is None or motion_frame.time_to_target < closest_motion_frame_time:
                    closest_motion_frame_time = motion_frame.time_to_target
                    frame_idx = idx
        if frame_idx == -1:
            self.get_logger().info("Recieved motion refernce exhausted for now")
            return
        motion_frame = self.motion_reference_msg.data[frame_idx]
        
        # update joint state for reference
        self.joint_state_msg.header.stamp = current_time.to_msg()
        self.joint_state_msg.name = []
        self.joint_state_msg.position = []
        self.joint_state_msg.velocity = []
        self.joint_state_msg.effort = []
        for i in range(len(motion_frame.dof_pos)):
            if motion_frame.dof_pos_mask[i]:
                self.joint_state_msg.name.append(self.robot_cfg.sim_dof_names[i])
                self.joint_state_msg.position.append(motion_frame.dof_pos[i])
                self.joint_state_msg.velocity.append(0.)
                self.joint_state_msg.effort.append(0.)
        self.joint_state_pub.publish(self.joint_state_msg)

        # publish base reference frame
        tf_msg = TransformStamped()
        tf_msg.header.stamp = current_time.to_msg()
        tf_msg.header.frame_id = "world"
        tf_msg.child_frame_id = "torso_link"
        tf_msg.transform.translation.x = motion_frame.position.x
        tf_msg.transform.translation.y = motion_frame.position.y
        tf_msg.transform.translation.z = motion_frame.position.z
        
        quat = quaternion.from_rotation_vector(np.array([
            motion_frame.axisangle.x,
            motion_frame.axisangle.y,
            motion_frame.axisangle.z,
        ]))
        tf_msg.transform.rotation.x = quat.x
        tf_msg.transform.rotation.y = quat.y
        tf_msg.transform.rotation.z = quat.z
        tf_msg.transform.rotation.w = quat.w
        self.tf_broadcaster.sendTransform(tf_msg)

        # publish interested link positions
        for i in range(len(motion_frame.link_pos)):
            link_pos = motion_frame.link_pos[i]
            tf_msg = TransformStamped()
            tf_msg.header.stamp = current_time.to_msg()
            tf_msg.header.frame_id = "torso_link"
            tf_msg.child_frame_id = "interested_link_" + str(i)
            tf_msg.transform.translation.x = link_pos.x
            tf_msg.transform.translation.y = link_pos.y
            tf_msg.transform.translation.z = link_pos.z
            tf_msg.transform.rotation.x = 0.
            tf_msg.transform.rotation.y = 0.
            tf_msg.transform.rotation.z = 0.
            tf_msg.transform.rotation.w = 1.
            self.tf_broadcaster.sendTransform(tf_msg)

async def rclpy_spin(node):
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0)
        await asyncio.sleep(1e-4)

async def main(args):
    rclpy.init()
    node = MotionReferencePublisher(args)
    
    tasks = [
        asyncio.ensure_future(node.motion_reference_publish()),
        asyncio.ensure_future(rclpy_spin(node)),
    ]
    
    loop = asyncio.get_event_loop()
    try:
        await asyncio.wait(tasks)
    except SystemExit:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        # shutdown should be called in the `motion_reference_publish` coroutine
        loop.stop()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bagdir", type=str, required=True, help="The directory which contains the .mcap file")
    parser.add_argument("--topic", type=str, default="/motion_reference", help="The topic to read from")
    parser.add_argument("--robot_class", type=str, default="G1_29Dof", help="The robot class to use")
    parser.add_argument("--debug_vis", action="store_true", help="Enable debug visualization")
    args = parser.parse_args()

    asyncio.run(main(args))
        