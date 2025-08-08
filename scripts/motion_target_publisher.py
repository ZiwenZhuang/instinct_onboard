import os

import numpy as np
import onnxruntime as ort
import quaternion
import rclpy
import yaml
from geometry_msgs.msg import Point, Quaternion
from rclpy.node import Node
from unitree_go.msg import WirelessController
from unitree_hg.msg import IMUState

from instinct_onboard import robot_cfgs, utils
from motion_target_msgs.msg import MotionFrame, MotionSequence


class MotionTargetPublisher(Node):
    @staticmethod
    def add_arguments(parser):
        """Add command line arguments for the node."""
        parser.add_argument(
            "--motion_file",
            type=str,
            help="Path to the motion file to be loaded.",
        )
        parser.add_argument(
            "--robot_class",
            type=str,
            default="G1_29Dof",
            help="Robot configuration to use.",
        )
        parser.add_argument("--logdir", type=str, help="Directory to load the motion logdir and link_of_interesets.")
        parser.add_argument(
            "--nonstop_at_exhausted",
            action="store_true",
            help="Publish always positive time_to_target even if the motion is exhausted.",
        )
        return parser

    def __init__(self, args):
        super().__init__("motion_target_publisher")

        self.args = args
        self.robot_cfg = getattr(robot_cfgs, args.robot_class, None)
        self.sim_joint_names = self.robot_cfg.sim_joint_names
        self.parse_config(args.logdir)
        self.load_fk_session(args.logdir)
        self.load_motion(args.motion_file)

    def parse_config(self, logdir: str):
        """Parse the configuration of the policy model"""
        with open(os.path.join(logdir, "params", "env.yaml")) as f:
            self.env_cfg = yaml.unsafe_load(f)
        self.motion_reference_config = self.env_cfg["scene"]["motion_reference"]
        # fill configs that will be frequently used in the node
        self.link_of_interests = self.motion_reference_config["link_of_interests"]  # order matters
        self.num_frames = self.motion_reference_config["num_frames"]  # num frames in the motion target message
        self.frame_interval_s = self.motion_reference_config["frame_interval_s"]
        self.motion_update_period_s = self.motion_reference_config["update_period"]

    def load_fk_session(self, logdir: str):
        """Load the forward kinematics model to compute the link positions."""
        # load the forward kinematics model to compute the link positions
        # NOTE: the link order should be the same as the link_of_interests
        ort_execution_providers = ort.get_available_providers()
        fk_model_path = os.path.join(logdir, "exported", "forward_kinematics.onnx")
        if not os.path.exists(fk_model_path):
            raise FileNotFoundError(f"Forward kinematics model {fk_model_path} does not exist.")
        self.fk_session = ort.InferenceSession(fk_model_path, providers=ort_execution_providers)
        self.fk_input_names = [input.name for input in self.fk_session.get_inputs()]
        self.fk_output_names = [output.name for output in self.fk_session.get_outputs()]
        self.get_logger().info(
            f"Loaded forward kinematics model from {fk_model_path} with inputs {self.fk_input_names} and outputs"
            f" {self.fk_output_names}."
        )

    def load_motion(self, motion_file: str):
        self.motion_file = motion_file
        if not os.path.exists(self.motion_file):
            raise FileNotFoundError(f"Motion file {self.motion_file} does not exist.")
        assert self.motion_file.endswith("_retargetted.npz"), "Motion file must end with '_retargetted.npz'."
        self.motion_data = np.load(self.motion_file, allow_pickle=True)
        self.motion_joint_names = (
            self.motion_data["joint_names"]
            if isinstance(self.motion_data["joint_names"], list)
            else self.motion_data["joint_names"].tolist()
        )
        joint_pos_ = self.motion_data["joint_pos"]
        retargetted_joints_to_output_joints_ids = [
            self.motion_joint_names.index(j_name) for j_name in self.sim_joint_names
        ]
        joint_pos = joint_pos_[:, retargetted_joints_to_output_joints_ids]
        root_trans = self.motion_data["base_pos_w"]
        root_quat = self.motion_data["base_quat_w"]
        self.num_joints = len(self.sim_joint_names)
        self.motion_buffer = {
            "framerate": self.motion_data["framerate"].item(),
            "joint_pos": joint_pos,  # shape (num_motion_frames, num_joints)
            "root_trans": root_trans,  # shape (num_motion_frames, 3)
            "root_quat": root_quat,  # shape (num_motion_frames, 4)
        }

        # run onnx session to compute the link positions and orientations (in base frame)
        link_pos = []
        link_quat = []
        for i in range(joint_pos.shape[0]):
            fk_inputs = {
                self.fk_input_names[0]: joint_pos[i : (i + 1)].astype(np.float32),
            }
            fk_outputs = self.fk_session.run(None, fk_inputs)
            link_pos.append(fk_outputs[0])  # shape (1, num_links, 3)
            link_quat.append(fk_outputs[1])  # shape (1, num_links, 4)
        self.motion_buffer["link_pos_b"] = np.concatenate(link_pos)  # shape (num_motion_frames, num_links, 3)
        self.motion_buffer["link_quat_b"] = np.concatenate(link_quat)  # shape (num_motion_frames, num_links, 4)
        self.get_logger().info("Motion file load done.")
        self.get_logger().info(
            f"Motion file {self.motion_file} loaded with {joint_pos.shape[0]} frames, total duration"
            f" {joint_pos.shape[0] / self.motion_buffer['framerate']:.2f} seconds with"
            f" {self.motion_buffer['framerate']} Hz."
        )

    """
    Functions with ROS handlers.
    """

    def start_ros_handlers(self):
        """Start the ROS handlers for the node."""
        self.robot_imu_subscriber = self.create_subscription(
            IMUState,
            "/secondary_imu",
            self.imu_callback,
            10,
        )
        self.wireless_controller_subscriber = self.create_subscription(
            WirelessController,
            "/wirelesscontroller",
            self.wireless_controller_callback,
            10,
        )
        self.motion_target_publisher = self.create_publisher(MotionSequence, "motion_target", 10)

        self.available_states = ["cold_start", "ready", "motion", "done"]
        self.current_state = "cold_start"
        self._motion_start_time = self.get_clock().now()
        self.get_logger().info(f"motion_update_period_s: {self.motion_update_period_s} seconds.")
        self.main_timer = self.create_timer(self.motion_update_period_s, self.main_loop_callback)

    def imu_callback(self, msg: IMUState):
        """Callback to update the IMU state for the initialization of the motion sequence."""
        self.imu_state = msg

    def wireless_controller_callback(self, msg: WirelessController):
        """Callback to update the wireless controller state for the initialization of the motion sequence."""
        self.wireless_controller_state = msg

    def main_loop_callback(self):
        """Main loop callback to publish the motion target."""
        assert (
            self.current_state in self.available_states
        ), f"Current state {self.current_state} is not in available states {self.available_states}."
        if self.current_state == "cold_start":
            self.get_logger().info("Waiting for the robot to be ready...", throttle_duration_sec=5)
            # Here you can check if the robot is ready to startupdate_heading_matching_quat the motion
            # For example, you can check if the IMU state and wireless controller state are available
            if hasattr(self, "imu_state") and hasattr(self, "wireless_controller_state"):
                self.update_heading_matching_quat()
                self.current_state = "ready"
                self.get_logger().info("Motion Target Publisher is ready. Transitioning to 'ready' state.")
        elif self.current_state == "ready":
            self.publish_starting_motion_target()
            if self.wireless_controller_state.keys & robot_cfgs.WirelessButtons.up:
                self.current_state = "motion"
                self.get_logger().info("Starting motion target publishing. Transitioning to 'motion' state.")
        elif self.current_state == "motion":
            self.publish_motion_target()
        elif self.current_state == "done":
            self.get_logger().info("In done state, shutting down the node.")
            raise SystemExit(0)

        if hasattr(self, "wireless_controller_state"):
            if (self.wireless_controller_state.keys & robot_cfgs.WirelessButtons.L2) or (
                self.wireless_controller_state.keys & robot_cfgs.WirelessButtons.R2
            ):
                self.get_logger().info("Stopping motion target publishing. Exiting.")
                self.current_state = "done"

    """
    Functions in each different state of the motion target publisher.
    """

    def update_heading_matching_quat(self):
        ref_quat_w = quaternion.from_float_array(self.motion_buffer["root_quat"][0])  # (w, x, y, z) order
        robot_quat_w = quaternion.from_float_array(
            [
                self.imu_state.quaternion[0],
                self.imu_state.quaternion[1],
                self.imu_state.quaternion[2],
                self.imu_state.quaternion[3],  # From Unitree IMUState message
            ]
        )
        # Assuming currently "heading_match_quat * ref_quat_w == robot_quat_w"
        from_ref_to_robot = robot_quat_w * ref_quat_w.conjugate()
        self._heading_matching_quat = utils.yaw_quat(from_ref_to_robot)

        self.get_logger().info("Updating headings for all motion frames.")
        self.motion_buffer["root_quat"] = quaternion.as_float_array(
            self._heading_matching_quat * quaternion.from_float_array(self.motion_buffer["root_quat"])
        )

    def publish_starting_motion_target(self):
        """Publish the starting motion target. Also set the motion start time as now."""
        self._motion_start_time = self.get_clock().now()

        frame_idx = np.arange(self.num_frames, dtype=int) * self.frame_interval_s * self.motion_buffer["framerate"]
        frame_idx = np.clip(
            np.floor(frame_idx).astype(int),
            0,
            self.motion_buffer["joint_pos"].shape[0] - 1,
        )
        root_trans_anchor = self.motion_buffer["root_trans"][0]
        root_quat_anchor = self.motion_buffer["root_quat"][0]
        self.publish_motion_by_frame_idx(
            frame_idxs=frame_idx,
            root_trans_anchor=root_trans_anchor,
            root_quat_anchor=root_quat_anchor,
            anchor_time=0.0,
        )

    def publish_motion_target(self):
        anchor_time = (self.get_clock().now() - self._motion_start_time).nanoseconds / 1e9  # seconds
        frame_idx_from_start = int(np.floor(anchor_time * self.motion_buffer["framerate"]))
        frame_time = (np.arange(self.num_frames) + 1) * self.frame_interval_s + anchor_time
        frame_idxs = np.clip(
            np.floor(frame_time * self.motion_buffer["framerate"]).astype(int),
            0,
            self.motion_buffer["joint_pos"].shape[0] - 1,
        )  # ensure we don't go out of bounds, shape (num_frames,)

        # Prepare the motion target message
        if frame_idx_from_start >= self.motion_buffer["joint_pos"].shape[0]:
            frame_idx_from_start = self.motion_buffer["joint_pos"].shape[0] - 1
        root_trans_anchor = self.motion_buffer["root_trans"][frame_idx_from_start]
        root_quat_anchor = self.motion_buffer["root_quat"][frame_idx_from_start]
        self.publish_motion_by_frame_idx(
            frame_idxs=frame_idxs,
            root_trans_anchor=root_trans_anchor,
            root_quat_anchor=root_quat_anchor,
            anchor_time=anchor_time,
        )

    def publish_motion_by_frame_idx(
        self,
        frame_idxs: np.ndarray,  # shape (num_frames,), type int
        root_trans_anchor: np.ndarray,  # shape (3,), type float
        root_quat_anchor: np.ndarray,  # shape (4,), type float, (w, x, y, z) order
        anchor_time: float,  # seconds, time of the anchor frame, w.r.t motion starting time.
    ):
        motion_target_msg = MotionSequence()
        motion_target_msg.joint_names = self.sim_joint_names
        motion_target_msg.link_names = self.link_of_interests
        frame_time = frame_idxs / self.motion_buffer["framerate"]
        for i, frame_idx in enumerate(frame_idxs):
            quat_w = self.motion_buffer["root_quat"][frame_idx]
            pos_offset_w = self.motion_buffer["root_trans"][frame_idx] - root_trans_anchor
            pos_b = quaternion.rotate_vectors(
                quaternion.from_float_array(root_quat_anchor).conjugate(), pos_offset_w
            ).astype(np.float32)
            pose_mask = [True] * 4

            joint_pos = self.motion_buffer["joint_pos"][frame_idx]  # joint_pos in urdf space, simulation order
            joint_pos_mask = [True] * self.num_joints  # Assuming all joints are valid

            link_pos = self.motion_buffer["link_pos_b"][frame_idx]
            link_pos_mask = [True] * len(self.link_of_interests)
            link_quat = self.motion_buffer["link_quat_b"][frame_idx]
            link_quat_mask = [True] * len(self.link_of_interests)
            assert len(link_pos) == len(self.link_of_interests), (
                f"Link positions length {len(link_pos)} does not match link of interests length"
                f" {len(self.link_of_interests)}."
            )

            # pack into MotionFrame
            motion_frame = MotionFrame()
            time_to_target = frame_time[i] - anchor_time  # seconds
            if time_to_target < 0:
                if self.args.nonstop_at_exhausted:
                    time_to_target = self.motion_update_period_s * (i + 1)
                else:
                    time_to_target = -1.0
            motion_frame.time_to_target = time_to_target
            motion_frame.pos_b.x = pos_b[0].item()
            motion_frame.pos_b.y = pos_b[1].item()
            motion_frame.pos_b.z = pos_b[2].item()
            motion_frame.quat_w.w = quat_w[0].item()
            motion_frame.quat_w.x = quat_w[1].item()
            motion_frame.quat_w.y = quat_w[2].item()
            motion_frame.quat_w.z = quat_w[3].item()
            motion_frame.pose_mask = pose_mask
            motion_frame.joint_pos = joint_pos.astype(np.float32).tolist()
            motion_frame.joint_pos_mask = joint_pos_mask
            for link_i in range(len(self.link_of_interests)):
                # link_name = self.link_of_interests[link_i]
                motion_frame.link_pos.append(
                    Point(
                        x=link_pos[link_i][0].item(),
                        y=link_pos[link_i][1].item(),
                        z=link_pos[link_i][2].item(),
                    )
                )
            motion_frame.link_pos_mask = link_pos_mask
            for link_i in range(len(self.link_of_interests)):
                link_name = self.link_of_interests[link_i]
                motion_frame.link_quat.append(
                    Quaternion(
                        w=link_quat[link_i][0].item(),
                        x=link_quat[link_i][1].item(),
                        y=link_quat[link_i][2].item(),
                        z=link_quat[link_i][3].item(),
                    )
                )
            motion_frame.link_quat_mask = link_quat_mask
            motion_target_msg.data.append(motion_frame)

        # Set the header for the motion target message and publish it
        motion_target_msg.header.stamp = self.get_clock().now().to_msg()
        motion_target_msg.header.frame_id = "base_link"
        self.motion_target_publisher.publish(motion_target_msg)

        # warn if no motion frames are valid
        if all(frame_time[i] - anchor_time < 0 for i in range(len(frame_time))):
            if self.args.nonstop_at_exhausted:
                self.get_logger().info("All motion frames are exhausted, but nonstop_at_exhausted is enabled. ")
            else:
                self.get_logger().info("All motion frames are exhausted, all published motion will be invalid.")


def main(args):
    rclpy.init()

    motion_target_publisher = MotionTargetPublisher(args)
    motion_target_publisher.start_ros_handlers()

    try:
        rclpy.spin(motion_target_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Motion Target Publisher for Unitree robots.")
    parser = MotionTargetPublisher.add_arguments(parser)

    args = parser.parse_args()
    main(args)
