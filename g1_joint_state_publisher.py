import rclpy
from rclpy.node import Node
from unitree_hg.msg import (
    WirelessController,
    LowState,
    # MotorState,
    # IMUState,
    LowCmd,
    # MotorCmd,
)

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState, PointCloud2, Image, CameraInfo

import numpy as np
import ros2_numpy as rnp


class RobotCfgs:
    class G1_29Dof:
        NUM_DOF = 29
        NUM_ACTIONS = 29
        dof_map = [
            15, 22, # shoulder pitch
            14, # waist pitch
            16, 23, # shoulder roll
            13, # waist roll
            17, 24, # shoulder yaw
            12, # waist yaw
            18, 25, # elbow
            0, 6, # hip pitch
            19, 26, # wrist roll
            1, 7, # hip roll
            20, 27, # wrist pitch
            2, 8, # hip yaw
            21, 28, # wrist yaw
            3, 9, # knee
            4, 10, # ankle pitch
            5, 11, # ankle roll
        ]
        dof_names = [ # NOTE: order matters. This list is the order in simulation.
            'left_shoulder_pitch_joint', #
            'right_shoulder_pitch_joint',
            'waist_pitch_joint',
            'left_shoulder_roll_joint', #
            'right_shoulder_roll_joint',
            'waist_roll_joint',
            'left_shoulder_yaw_joint', #
            'right_shoulder_yaw_joint',
            'waist_yaw_joint',
            'left_elbow_joint', #
            'right_elbow_joint',
            'left_hip_pitch_joint',
            'right_hip_pitch_joint',
            'left_wrist_roll_joint',
            'right_wrist_roll_joint',
            'left_hip_roll_joint', #
            'right_hip_roll_joint',
            'left_wrist_pitch_joint',
            'right_wrist_pitch_joint',
            'left_hip_yaw_joint',
            'right_hip_yaw_joint',
            'left_wrist_yaw_joint', #
            'right_wrist_yaw_joint',
            'left_knee_joint',
            'right_knee_joint',
            'left_ankle_pitch_joint', #
            'right_ankle_pitch_joint',
            'left_ankle_roll_joint',
            'right_ankle_roll_joint',
        ]
        dof_signs = [
            1, 1, -1,
            1, 1, -1,
            1, 1, -1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ]
        joint_limits_high = [
            2.6704, 2.6704, 0.5200,
            2.2515, 1.5882, 0.5200,
            2.6180, 2.6180, 2.6180,
            2.0944, 2.0944, 2.8798, 2.8798, 1.9722, 1.9722,
            2.9671, 0.5236, 1.6144, 1.6144, 2.7576, 2.7576,
            1.6144, 1.6144, 2.8798, 2.8798,
            0.5236, 0.5236, 0.2618, 0.2618,
        ]
        joint_limits_low = [
            -3.0892, -3.0892, -0.5200,
            -1.5882, -2.2515, -0.5200,
            -2.6180, -2.6180, -2.6180,
            -1.0472, -1.0472, -2.5307, -2.5307, -1.9722, -1.9722,
            -0.5236, -2.9671, -1.6144, -1.6144, -2.7576, -2.7576,
            -1.6144, -1.6144, -0.0873, -0.0873,
            -0.8727, -0.8727, -0.2618, -0.2618,
        ]
        torque_limits = [ # from urdf and in simulation order
            25, 25, 50,
            25, 25, 50,
            25, 25, 88,
            25, 25, 88, 88, 25, 25,
            88, 88, 5, 5, 88, 88,
            5, 5, 139, 139,
            50, 50, 50, 50,
        ]
        turn_on_motor_mode = [0x01] * 29
        mode_pr = 0
        """ please check this value from
            https://support.unitree.com/home/zh/G1_developer/basic_services_interface
            https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description
        """


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

        self.robot_class = getattr(RobotCfgs, args.robot_model)

        self.low_state_sub = self.create_subscription(
            LowState,
            '/lowstate',
            self.low_state_callback,
            1,
        )

    def low_state_callback(self, msg):
        self.get_logger().info("low_state revieved", once=True)
        self.joint_state.header.stamp = self.get_clock().now().to_msg()
        self.joint_state.name = []
        self.joint_state.position = []
        self.joint_state.velocity = []
        self.joint_state.effort = []
        for i in range(len(self.robot_class.dof_names)):
            self.joint_state.name.append(self.robot_class.dof_names[i])
            self.joint_state.position.append(
                msg.motor_state[self.robot_class.dof_map[i]].q * self.robot_class.dof_signs[i],
            )
            self.joint_state.velocity.append(
                msg.motor_state[self.robot_class.dof_map[i]].dq * self.robot_class.dof_signs[i],
            )
            self.joint_state.effort.append(
                msg.motor_state[self.robot_class.dof_map[i]].tau_est * self.robot_class.dof_signs[i],
            )
        self.joint_state_pub.publish(self.joint_state)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = "world"
        tf_msg.child_frame_id = "torso_link"
        tf_msg.transform.translation.x = 0.
        tf_msg.transform.translation.y = 0.
        tf_msg.transform.translation.z = 0.
        tf_msg.transform.rotation.x = msg.imu_state.quaternion[1].item()
        tf_msg.transform.rotation.y = msg.imu_state.quaternion[2].item()
        tf_msg.transform.rotation.z = msg.imu_state.quaternion[3].item()
        tf_msg.transform.rotation.w = msg.imu_state.quaternion[0].item()
        self.tf_broadcaster.sendTransform(tf_msg)


def main(args):
    rclpy.init()
    joint_state_node = UnitreeROS2hgJointState(args)
    rclpy.spin(joint_state_node)
    joint_state_node.destory_node()
    rclpy.shutdown()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--robot_model", type=str, default="G1_29Dof",
        help="The robot model to select with corresponding configs (to match the simulation).",
    )

    args = parser.parse_args()
    main(args)
