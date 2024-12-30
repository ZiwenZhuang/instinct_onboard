import os, sys
import select
import asyncio

import rclpy
from rclpy.node import Node
import rclpy.time
import rosbag2_py
from rclpy.serialization import deserialize_message
# from rosidl_runtime_py.utilities import get_message

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, PoseArray
from sensor_msgs.msg import JointState

from unitree_go.msg import WirelessController
from unitree_hg.msg import IMUState

from motion_reference_msgs.msg import MotionReference, MotionFrame

import numpy as np
import quaternion # from numpy-quaternion
import robot_cfgs

def normalize_quat(quat: quaternion) -> quaternion:
    """Normalize the quaternion.

    Args:
        quat: The orientation in (w, x, y, z). Not Batched.

    Returns:
        A normalized quaternion, with w > 0.
    """
    quat = quat / np.linalg.norm(quaternion.as_float_array(quat), axis=-1).clip(min=1e-6)
    quat = quat * np.sign(quaternion.as_float_array(quat)[..., 0])
    return quat

def yaw_quat(quat: quaternion) -> quaternion:
    """Extract the yaw component of a quaternion.

    Args:
        quat: The orientation in (w, x, y, z). Not Batched.

    Returns:
        A quaternion with only yaw component.
    """
    qw = quat.w
    qx = quat.x
    qy = quat.y
    qz = quat.z
    yaw = np.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw = np.zeros(4)
    quat_yaw[3] = np.sin(yaw / 2)
    quat_yaw[0] = np.cos(yaw / 2)
    return normalize_quat(quaternion.as_quat_array(quat_yaw))

def inv_quat(quat: quaternion) -> quaternion:
    quat_norm = np.linalg.norm(quaternion.as_float_array(quat)).clip(min=1e-6)
    return quat.conjugate() / (quat_norm ** 2)

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

        self.imu_state_sub = self.create_subscription(
            IMUState,
            "/secondary_imu",
            self.imu_state_callback,
            1,
        )
        self.wireless_sub = self.create_subscription(
            WirelessController,
            "/wirelesscontroller",
            self.wireless_callback,
            1,
        )

        # create some buffers incase the rosbag file is not in order
        self._motion_reference_buffer = []
        self._pose_w_buffer = []

        # wait for imu_state_buffer to be filled
        while not hasattr(self, "imu_state_buffer") and rclpy.ok():
            rclpy.spin_once(self)

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
        while rclpy.ok():
            # check exit conditions from wireless controller
            if hasattr(self, "wireless_buffer") and \
                ((self.wireless_buffer.keys & robot_cfgs.WirelessButtons.R2) or (self.wireless_buffer.keys & robot_cfgs.WirelessButtons.L2)):
                self.get_logger().info("Received R2 or L2, Program Exit")
                rclpy.shutdown()
                return
            # check continue conditions from wireless controller
            if hasattr(self, "wireless_buffer") and (self.wireless_buffer.keys & robot_cfgs.WirelessButtons.L1):
                self.get_logger().info("Received L1, Continue to policy stage")
                break
            
            # check exit/condinue conditions from keyboard input
            if select.select([sys.stdin], [], [], 0)[0]:
                # async input
                input_str = sys.stdin.readline().strip()
                if input_str == "":
                    break
                elif input_str == "q":
                    self.get_logger().info("Received q, Program Exit")
                    rclpy.shutdown()
                    return
            
            await asyncio.sleep(1e-3)

        while self.publish_a_frame() and rclpy.ok():
            duration = self.rosbag_msg_time - self.prev_rosbag_msg_time
            sleep_time_f = (duration).nanoseconds / 1e9
            await asyncio.sleep(sleep_time_f)
            self.prev_rosbag_msg_time = self.rosbag_msg_time
        
        self.get_logger().info("Finished publishing all frames")
        rclpy.shutdown()

    def publish_a_frame(self) -> bool:
        msgs = self.pop_next_msg_frame()
        if msgs is not None:
            _, motion_reference_msg, _ = msgs[self.args.topic]
            # Get the rosbag time using timestamp or header.stamp
            if False:
                timestamp_sec = int(timestamp / 1e9); timestamp_nsec = int(timestamp % 1e9)
                self.rosbag_msg_time = rclpy.time.Time(seconds=timestamp_sec, nanoseconds=timestamp_nsec)
            else:
                self.rosbag_msg_time = rclpy.time.Time.from_msg(motion_reference_msg.header.stamp)

            # update the pose reference based on robot's low state and global rotation reference
            if hasattr(self, "imu_state_buffer"):
                _, pose_w_msg, _ = msgs[self.args.pose_w_topic]
                pos_offset_w = np.array([[
                    pose_w.position.x,
                    pose_w.position.y,
                    pose_w.position.z,
                ] for pose_w in pose_w_msg.poses]) # (N, 3)
                ref_quat_w = quaternion.as_quat_array(np.array([[
                    pose_w.orientation.w,
                    pose_w.orientation.x,
                    pose_w.orientation.y,
                    pose_w.orientation.z,
                ] for pose_w in pose_w_msg.poses])) # (N, 4)
                root_quat_w = quaternion.as_quat_array(np.array([
                    self.imu_state_buffer.quaternion[0],
                    self.imu_state_buffer.quaternion[1],
                    self.imu_state_buffer.quaternion[2],
                    self.imu_state_buffer.quaternion[3],
                ])) # (4,)
                root_quat_w_inv = inv_quat(root_quat_w)
                if not hasattr(self, "match_heading_quat_w"):
                    # The first frame, update the robot's heading
                    root_heading_quat_w = yaw_quat(root_quat_w)
                    ref_heading_quat_w = yaw_quat(ref_quat_w[0])
                    self.match_heading_quat_w = normalize_quat(root_heading_quat_w * inv_quat(ref_heading_quat_w))
                # update the heading based on the initial heading of the robot.
                ref_quat_w = normalize_quat(self.match_heading_quat_w * ref_quat_w)
                pos_offset_w = quaternion.rotate_vectors(
                    self.match_heading_quat_w,
                    pos_offset_w,
                ) # (N, 3)
                # compute the pose reference in base frame
                pos_offset_b = quaternion.rotate_vectors(
                    root_quat_w_inv,
                    pos_offset_w,
                ) # (N, 3)
                quat_b = normalize_quat(root_quat_w_inv * ref_quat_w) # (N, 4)
                axisang_b = quaternion.as_rotation_vector(quat_b) # (N, 3)
                for frame_idx in range(len(motion_reference_msg.data)):
                    motion_frame = motion_reference_msg.data[frame_idx]
                    motion_frame.position.x = pos_offset_b[frame_idx, 0]
                    motion_frame.position.y = pos_offset_b[frame_idx, 1]
                    motion_frame.position.z = pos_offset_b[frame_idx, 2]
                    motion_frame.axisangle.x = axisang_b[frame_idx, 0]
                    motion_frame.axisangle.y = axisang_b[frame_idx, 1]
                    motion_frame.axisangle.z = axisang_b[frame_idx, 2]
                
            # update the real pulish time.
            self.msg_publish_time = self.get_clock().now()
            # update header timestamp
            motion_reference_msg.header.stamp = self.msg_publish_time.to_msg()
            self.motion_reference_msg = motion_reference_msg
            self.motion_reference_pub.publish(motion_reference_msg)
            self.get_logger().info("Published motion reference message at {:.3f}s".format(self.msg_publish_time.nanoseconds / 1e9))
            return True
        else:
            self.get_logger().info("No more messages to publish")
            delattr(self, "bag_reader")
            return False

    def pop_next_msg_frame(self):
        msg_frame_flag = np.array([False, False]) # [motion_reference, pose_w]
        # Only when both messages are collected, return the message.
        while not msg_frame_flag.all() and self.bag_reader.has_next():
            topic, data, timestamp = self.bag_reader.read_next()
            try:
                if topic == self.args.topic:
                    msg = deserialize_message(data, MotionReference)
                    self._motion_reference_buffer.append((topic, msg, timestamp))
                    msg_frame_flag[0] = True
                elif topic == self.args.pose_w_topic:
                    msg = deserialize_message(data, PoseArray)
                    self._pose_w_buffer.append((topic, msg, timestamp))
                    msg_frame_flag[1] = True
                else:
                    self.get_logger().warn("Unexpected topic: " + str(topic))
                    return None
            except Exception as e:
                self.get_logger().error("Failed to deserialize message: " + str(e))
                return None
        if len(self._motion_reference_buffer) == 0 and len(self._pose_w_buffer) == 0:
            # no more message to publish
            return None
        # check if both buffer has the same length and the same timestamp
        # Maybe other cases could be handled, but for now, just accept the same length and timestamp.
        if len(self._motion_reference_buffer) != len(self._pose_w_buffer):
            self.get_logger().error("Buffer length mismatch: {:d} vs {:d}".format(len(self._motion_reference_buffer), len(self._pose_w_buffer)))
            return None
        if len(self._motion_reference_buffer) > 0 and len(self._pose_w_buffer) > 0 and \
            abs(self._motion_reference_buffer[0][2] - self._pose_w_buffer[0][2]) > 1e6:
            self.get_logger().error("Timestamp mismatch: {:d} vs {:d}".format(self._motion_reference_buffer[0][2], self._pose_w_buffer[0][2]))
            return None
        return {
            self.args.topic: self._motion_reference_buffer.pop(0),
            self.args.pose_w_topic: self._pose_w_buffer.pop(0),
        }

    def imu_state_callback(self, msg: IMUState):
        self.imu_state_buffer = msg

    def wireless_callback(self, msg: WirelessController):
        self.wireless_buffer = msg
        
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
            # self.get_logger().info("Recieved motion refernce exhausted for now")
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
                self.joint_state_msg.name.append(self.robot_cfg.sim_joint_names[i])
                self.joint_state_msg.position.append(motion_frame.dof_pos[i])
                self.joint_state_msg.velocity.append(0.)
                self.joint_state_msg.effort.append(0.)
        self.joint_state_pub.publish(self.joint_state_msg)

        # if has low_state, publish orientation in world frame
        if hasattr(self, "imu_state_buffer"):
            tf_msg = TransformStamped()
            tf_msg.header.stamp = current_time.to_msg()
            tf_msg.header.frame_id = "world"
            tf_msg.child_frame_id = "torso_link"
            tf_msg.transform.translation.x = 0.
            tf_msg.transform.translation.y = 0.
            tf_msg.transform.translation.z = 0.
            tf_msg.transform.rotation.w = float(self.imu_state_buffer.quaternion[0])
            tf_msg.transform.rotation.x = float(self.imu_state_buffer.quaternion[1])
            tf_msg.transform.rotation.y = float(self.imu_state_buffer.quaternion[2])
            tf_msg.transform.rotation.z = float(self.imu_state_buffer.quaternion[3])
            self.tf_broadcaster.sendTransform(tf_msg)

        # publish base reference frame
        tf_msg = TransformStamped()
        tf_msg.header.stamp = current_time.to_msg()
        tf_msg.header.frame_id = "torso_link"
        tf_msg.child_frame_id = "base_pose_ref"
        tf_msg.transform.translation.x = motion_frame.position.x
        tf_msg.transform.translation.y = motion_frame.position.y
        tf_msg.transform.translation.z = motion_frame.position.z
        quat = quaternion.from_rotation_vector(np.array([
            motion_frame.axisangle.x,
            motion_frame.axisangle.y,
            motion_frame.axisangle.z,
        ]))
        tf_msg.transform.rotation.w = quat.w
        tf_msg.transform.rotation.x = quat.x
        tf_msg.transform.rotation.y = quat.y
        tf_msg.transform.rotation.z = quat.z
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
            tf_msg.transform.rotation.w = 1.
            tf_msg.transform.rotation.x = 0.
            tf_msg.transform.rotation.y = 0.
            tf_msg.transform.rotation.z = 0.
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
    parser.add_argument("--pose_w_topic", type=str, default="/global_rotation_reference", help="The topic to read the world pose from")
    parser.add_argument("--robot_class", type=str, default="G1_29Dof", help="The robot class to use")
    parser.add_argument("--debug_vis", action="store_true", help="Enable debug visualization")
    args = parser.parse_args()

    asyncio.run(main(args))
        