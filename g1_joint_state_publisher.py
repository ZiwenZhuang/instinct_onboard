import rclpy
from rclpy.node import Node
from unitree_hg.msg import (
    WirelessController,
    LowState,
    # MotorState,
    IMUState,
    LowCmd,
    # MotorCmd,
)

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState, PointCloud2, Image, CameraInfo

import numpy as np
import ros2_numpy as rnp

import robot_cfgs

class UnitreeROS2hgJointState(Node):
    def __init__(self, args):
        super().__init__('unitree_ros2_joint_state')
        
        self.joint_state = JointState()
        self.joint_state_pub = self.create_publisher(
            JointState,
            'joint_states',
            1,
        )
        self.tf_broadcaster = TransformBroadcaster(self)

        self.robot_class = getattr(robot_cfgs, args.robot_class)

        self.low_state_sub = self.create_subscription(
            LowState,
            args.topic,
            self.low_state_callback,
            1,
        )
        self.torso_imu_sub = self.create_subscription(
            IMUState,
            args.torso_imu_topic,
            self.torso_imu_callback,
            1,
        )

    def low_state_callback(self, msg):
        self.get_logger().info("low_state revieved", once=True)
        self.joint_state.header.stamp = self.get_clock().now().to_msg()
        self.joint_state.name = []
        self.joint_state.position = []
        self.joint_state.velocity = []
        self.joint_state.effort = []
        for i in range(len(self.robot_class.sim_joint_names)):
            self.joint_state.name.append(self.robot_class.sim_joint_names[i])
            self.joint_state.position.append(
                msg.motor_state[self.robot_class.joint_map[i]].q * self.robot_class.joint_signs[i],
            )
            self.joint_state.velocity.append(
                msg.motor_state[self.robot_class.joint_map[i]].dq * self.robot_class.joint_signs[i],
            )
            self.joint_state.effort.append(
                msg.motor_state[self.robot_class.joint_map[i]].tau_est * self.robot_class.joint_signs[i],
            )
        self.joint_state_pub.publish(self.joint_state)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = "world"
        tf_msg.child_frame_id = "torso_link"
        tf_msg.transform.translation.x = 0.
        tf_msg.transform.translation.y = 0.
        tf_msg.transform.translation.z = 0.
        if hasattr(self, "torso_imu_buffer"):
            tf_msg.transform.rotation.x = self.torso_imu_buffer.quaternion[1].item()
            tf_msg.transform.rotation.y = self.torso_imu_buffer.quaternion[2].item()
            tf_msg.transform.rotation.z = self.torso_imu_buffer.quaternion[3].item()
            tf_msg.transform.rotation.w = self.torso_imu_buffer.quaternion[0].item()
        else:
            tf_msg.transform.rotation.x = msg.imu_state.quaternion[1].item()
            tf_msg.transform.rotation.y = msg.imu_state.quaternion[2].item()
            tf_msg.transform.rotation.z = msg.imu_state.quaternion[3].item()
            tf_msg.transform.rotation.w = msg.imu_state.quaternion[0].item()
        self.tf_broadcaster.sendTransform(tf_msg)

    def torso_imu_callback(self, msg):
        self.get_logger().info("torso_imu revieved", once=True)
        self.torso_imu_buffer = msg

def main(args):
    rclpy.init()
    joint_state_node = UnitreeROS2hgJointState(args)
    rclpy.spin(joint_state_node)
    joint_state_node.destory_node()
    rclpy.shutdown()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--robot_class", type=str, default="G1_29Dof",
        help="The robot model to select with corresponding configs (to match the simulation).",
    )
    parser.add_argument("--topic", type=str, default="/lowstate",)
    parser.add_argument("--torso_imu_topic", type=str, default="/secondary_imu",)
    parser.add_argument("--from_cmd", action="store_true", default=False,
        help="Use this flag to read the arguments from the command line.",)

    args = parser.parse_args()
    main(args)
