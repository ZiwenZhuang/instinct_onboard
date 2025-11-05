from __future__ import annotations

import math
import os

import cv2
import numpy as np
import onnxruntime as ort
import quaternion
import ros2_numpy as rnp
from geometry_msgs.msg import TransformStamped
from rclpy.publisher import Publisher
from sensor_msgs.msg import Image, PointCloud2, PointField
from tf2_ros import StaticTransformBroadcaster

from instinct_onboard.agents.base import ColdStartAgent, OnboardAgent
from instinct_onboard.normalizer import Normalizer
from instinct_onboard.ros_nodes.ros_real import Ros2Real
from instinct_onboard.utils import (
    inv_quat,
    quat_rotate_inverse,
    quat_to_tan_norm_batch,
    yaw_quat,
)


class TrackerAgent(OnboardAgent):
    """Different from ShadowingAgent, this agent reads the motion file directly and does not listen from the
    motion sequence topic. And we assume the network is just a MLP.
    """

    def __init__(
        self,
        logdir: str,
        motion_file: str,  # retargetted motion file
        ros_node: Ros2Real,
    ):
        super().__init__(logdir, ros_node)
        self.ort_sessions = dict()
        self._parse_obs_config()
        self._parse_action_config()
        self._load_models()
        self._load_motion(motion_file)

    def _load_models(self):
        """Load the ONNX model for the agent."""
        # load ONNX models
        ort_execution_providers = ort.get_available_providers()
        actor_path = os.path.join(self.logdir, "exported", "actor.onnx")
        self.ort_sessions["actor"] = ort.InferenceSession(actor_path, providers=ort_execution_providers)
        print(f"Loaded ONNX models from {self.logdir}")
        # load the normalizer
        normalizer_path = os.path.join(self.logdir, "exported", "policy_normalizer.npz")
        self.normalizer = Normalizer(load_path=normalizer_path)

    def _load_motion(self, motion_file: str):
        """Load the motion file."""
        self.motion_data = np.load(motion_file, allow_pickle=True)
        self.motion_framerate = self.motion_data["framerate"].item()

        # get the motion joint names and convert to robot joint names
        motion_joint_names = self.motion_data["joint_names"].tolist()
        robot_joint_names = self.ros_node.sim_joint_names
        motion_joint_to_robot_joint_ids = [motion_joint_names.index(j_name) for j_name in robot_joint_names]

        self.motion_joint_names = [motion_joint_names[i] for i in motion_joint_to_robot_joint_ids]
        self.motion_joint_pos = self.motion_data["joint_pos"][:, motion_joint_to_robot_joint_ids]
        joint_pos_ = np.concatenate([self.motion_joint_pos[0:1], self.motion_joint_pos])
        self.motion_joint_vel = (joint_pos_[1:] - joint_pos_[:-1]) * self.motion_framerate
        self.motion_base_pos = self.motion_data["base_pos_w"]
        self.motion_base_quat = self.motion_data["base_quat_w"]
        self.motion_total_num_frames = self.motion_data["joint_pos"].shape[0]

        # prepare the frame indices (offset w.r.t current cursor)
        self.motion_num_frames = self.cfg["scene"]["motion_reference"][
            "num_frames"
        ]  # the num of frames to output as reference.
        self.motion_frame_indices_offset = np.arange(self.motion_num_frames).astype(float)
        if self.cfg["scene"]["motion_reference"]["data_start_from"] == "one_frame_interval":
            self.motion_frame_indices_offset += 1
        self.motion_frame_indices_offset *= (
            self.cfg["scene"]["motion_reference"]["frame_interval_s"] * self.motion_framerate
        )
        self.motion_frame_indices_offset = self.motion_frame_indices_offset.astype(int)

        self.motion_cursor_idx = 0

    def reset(self):
        """Reset the agent state and the rosbag reader."""
        super().reset()
        self.motion_cursor_idx = 0

    def step(self):
        """Perform a single step of the agent."""
        self.motion_cursor_idx += 1
        done = self.motion_cursor_idx >= self.motion_total_num_frames - 1
        self.motion_cursor_idx = (
            self.motion_cursor_idx
            if self.motion_cursor_idx < self.motion_total_num_frames
            else self.motion_total_num_frames - 1
        )
        obs = self._get_observation()
        normalized_obs = self.normalizer.normalize(obs).astype(np.float32)[None, :]
        actor_input_name = self.ort_sessions["actor"].get_inputs()[0].name
        action = self.ort_sessions["actor"].run(None, {actor_input_name: normalized_obs})[0]
        action = action.reshape(-1)
        return action, done

    def match_to_current_heading(self):
        """Match the motion's 0-th frame to the current heading."""
        root_quat_w = quaternion.from_float_array(self.ros_node._get_quat_w_obs())  # (,) quaternion
        quat_w_ref = quaternion.from_float_array(self.motion_base_quat[0])  # (,) quaternion
        quat_err = root_quat_w * inv_quat(quat_w_ref)  # (,) quaternion
        heading_err_quat = yaw_quat(quat_err)  # (,) quaternion
        heading_err_quat_ = np.stack([heading_err_quat for _ in range(len(self.motion_base_quat))], axis=0)  # (N, 4)

        # update the base_quat_w for each frame
        motion_quats = quaternion.from_float_array(self.motion_base_quat)  # (N,) quaternion
        updated_quats = heading_err_quat_ * motion_quats  # broadcasts to (N,)
        self.motion_base_quat = quaternion.as_float_array(updated_quats)  # (N, 4)

        # update the base_pos_w for each frame
        current_pos_w = self.motion_base_pos[0]  # (3,)
        rel_pos = self.motion_base_pos - self.motion_base_pos[0:1]  # (N, 3)
        rotated_rel_pos = quaternion.rotate_vectors(heading_err_quat, rel_pos)  # (N, 3)
        self.motion_base_pos = rotated_rel_pos + current_pos_w[None, :]  # (N, 3)

    """
    Agent specific observation functions for TrackerAgent.
    """

    def _get_joint_pos_ref_command_cmd_obs(self):
        frame_indices = self.motion_frame_indices_offset + self.motion_cursor_idx
        frame_indices = frame_indices.clip(max=self.motion_total_num_frames - 1)
        return self.motion_joint_pos[frame_indices] - self.default_joint_pos[None, :]

    def _get_joint_vel_ref_command_cmd_obs(self):
        frame_indices = self.motion_frame_indices_offset + self.motion_cursor_idx
        frame_indices = frame_indices.clip(max=self.motion_total_num_frames - 1)
        return self.motion_joint_vel[frame_indices]

    def _get_position_b_ref_command_cmd_obs(self):
        """Return the future position reference in current motion reference's base frame."""
        frame_indices = self.motion_frame_indices_offset + self.motion_cursor_idx
        frame_indices = frame_indices.clip(max=self.motion_total_num_frames - 1)
        current_motion_base_pos = self.motion_base_pos[self.motion_cursor_idx : self.motion_cursor_idx + 1]
        current_motion_base_quat = self.motion_base_quat[self.motion_cursor_idx]  # (4,)
        future_motion_base_pos = self.motion_base_pos[frame_indices]
        future_motion_base_pos_b = quat_rotate_inverse(
            quaternion.from_float_array(current_motion_base_quat), future_motion_base_pos - current_motion_base_pos
        )
        return future_motion_base_pos_b  # (num_frames, 3)

    def _get_rotation_ref_command_cmd_obs(self):
        """
        Return the future rotation reference in current robot's base frame.
        """
        frame_indices = self.motion_frame_indices_offset + self.motion_cursor_idx
        frame_indices = frame_indices.clip(max=self.motion_total_num_frames - 1)
        current_robot_base_quat = self.ros_node._get_quat_w_obs()[None, :]  # (1, 4)
        future_motion_base_quat = self.motion_base_quat[frame_indices]
        future_motion_base_quat_b = inv_quat(
            quaternion.from_float_array(current_robot_base_quat)
        ) * quaternion.from_float_array(future_motion_base_quat)
        return quat_to_tan_norm_batch(future_motion_base_quat_b)  # (num_frames, 6)

    def get_cold_start_agent(self, startup_step_size: float = 0.2, kpkd_factor: float = 1.0) -> ColdStartAgent:
        """Create a ColdStartAgent with joint_target_pos set to the 0-th frame of the motion."""
        joint_target_pos = self.motion_joint_pos[0].copy()
        return ColdStartAgent(
            startup_step_size=startup_step_size,
            ros_node=self.ros_node,
            joint_target_pos=joint_target_pos,
            action_scale=self.action_scale,  # Note: passing action_offset here sets _action_offset in ColdStartAgent due to parameter naming in init
            action_offset=self.action_offset,  # passing action_scale here sets _action_scale
            p_gains=self.p_gains * kpkd_factor,
            d_gains=self.d_gains * kpkd_factor,
        )


class PerceptiveTrackerAgent(TrackerAgent):

    def __init__(self, *args, depth_vis: bool = False, pointcloud_vis: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth_vis = depth_vis
        if self.depth_vis:
            self.debug_depth_publisher = self.ros_node.create_publisher(Image, "/realsense/depth_image", 10)
        else:
            self.debug_depth_publisher = None
        self.pointcloud_vis = pointcloud_vis
        if self.pointcloud_vis:
            # Publish static TF
            self.static_tf_broadcaster = StaticTransformBroadcaster(self.ros_node)
            t = TransformStamped()
            t.header.stamp = self.ros_node.get_clock().now().to_msg()
            t.header.frame_id = "torso_link"
            t.child_frame_id = "d435_depth_link"
            t.transform.translation.x = 0.04764571478 + 0.0039635 - 0.0042 * math.cos(math.radians(48))
            t.transform.translation.y = 0.015
            t.transform.translation.z = 0.46268178553 - 0.044 + 0.0042 * math.sin(math.radians(48)) + 0.016
            t.transform.rotation.w = math.cos(math.radians(0.5) / 2) * math.cos(math.radians(48) / 2)
            t.transform.rotation.x = math.sin(math.radians(0.5) / 2)
            t.transform.rotation.y = math.sin(math.radians(48) / 2)
            t.transform.rotation.z = 0.0
            self.static_tf_broadcaster.sendTransform(t)

            self.debug_pointcloud_publisher = self.ros_node.create_publisher(PointCloud2, "/realsense/pointcloud", 10)
        else:
            self.debug_pointcloud_publisher = None

    def _load_models(self):
        super()._load_models()
        ort_execution_providers = ort.get_available_providers()
        depth_image_encoder_path = os.path.join(self.logdir, "exported", "0-depth_image.onnx")
        self.ort_sessions["depth_image_encoder"] = ort.InferenceSession(
            depth_image_encoder_path, providers=ort_execution_providers
        )

    def _parse_obs_config(self):
        super()._parse_obs_config()
        # add depth image cropping and normalization configs
        sim_resolution_before_crop = (
            self.cfg["scene"]["camera"]["pattern_cfg"]["width"],
            self.cfg["scene"]["camera"]["pattern_cfg"]["height"],
        )
        sim_crop_region = self.cfg["scene"]["camera"]["noise_pipeline"]["crop_and_resize"][
            "crop_region"
        ]  # up, down, left, right
        real_resolution = self.ros_node.rs_resolution  # (width, height)
        real_crop_region = (
            int(sim_crop_region[0] * real_resolution[1] / sim_resolution_before_crop[1]),  # up
            int(sim_crop_region[1] * real_resolution[1] / sim_resolution_before_crop[1]),  # down
            int(sim_crop_region[2] * real_resolution[0] / sim_resolution_before_crop[0]),  # left
            int(sim_crop_region[3] * real_resolution[0] / sim_resolution_before_crop[0]),  # right
        )
        self.depth_image_crop_region = real_crop_region  # (up, down, left, right)
        self.depth_image_final_resolution = self.cfg["scene"]["camera"]["noise_pipeline"]["crop_and_resize"][
            "resize_shape"
        ]  # (height, width)
        self.depth_image_clip_range = self.cfg["scene"]["camera"]["noise_pipeline"]["normalize"][
            "depth_range"
        ]  # (min, max)
        self.depth_image_shall_normalize = self.cfg["scene"]["camera"]["noise_pipeline"]["normalize"]["normalize"]
        self.depth_image_normalized_output_range = self.cfg["scene"]["camera"]["noise_pipeline"]["normalize"][
            "output_range"
        ]
        if "gaussian_blur_noise" in self.cfg["scene"]["camera"]["noise_pipeline"]:
            self.depth_image_gaussian_blur_kernel_size = self.cfg["scene"]["camera"]["noise_pipeline"][
                "gaussian_blur_noise"
            ]["kernel_size"]
            self.depth_image_gaussian_blur_sigma = self.cfg["scene"]["camera"]["noise_pipeline"]["gaussian_blur_noise"][
                "sigma"
            ]
        else:
            self.depth_image_gaussian_blur_kernel_size = None
            self.depth_image_gaussian_blur_sigma = None

    def reset(self):
        super().reset()
        if not hasattr(self, "depth_image_slice"):
            self.depth_image_slice = self._get_obs_slice("depth_image")

    def step(self):
        self.motion_cursor_idx += 1
        done = self.motion_cursor_idx >= self.motion_total_num_frames - 1
        self.motion_cursor_idx = (
            self.motion_cursor_idx
            if self.motion_cursor_idx < self.motion_total_num_frames
            else self.motion_total_num_frames - 1
        )
        obs = self._get_observation()
        normalized_obs = self.normalizer.normalize(obs).astype(np.float32)[None, :]
        depth_image = normalized_obs[:, self.depth_image_slice].reshape(
            1, 1, *self.depth_image_final_resolution
        )  # (1, 1, height, width)
        depth_image_encoder_input_name = self.ort_sessions["depth_image_encoder"].get_inputs()[0].name
        depth_embedding = self.ort_sessions["depth_image_encoder"].run(
            None, {depth_image_encoder_input_name: depth_image}
        )[0]
        actor_input = np.concatenate(
            [
                normalized_obs[:, : self.depth_image_slice.start],
                normalized_obs[:, self.depth_image_slice.stop :],
                depth_embedding,
            ],
            axis=-1,
        )  # (1, dim)
        actor_input_name = self.ort_sessions["actor"].get_inputs()[0].name
        action = self.ort_sessions["actor"].run(None, {actor_input_name: actor_input})[0]
        action = action.reshape(-1)

        if self.debug_depth_publisher is not None:
            # NOTE: the +5.0 is a empirical value to ensure the normalized obs is not negative.
            # Not using normalizer's value is to prevent further visualization code bugs.
            depth_image_msg_data = np.asanyarray(
                obs[self.depth_image_slice].reshape(*self.depth_image_final_resolution) * 255 * 2,
                dtype=np.uint16,
            )
            depth_image_msg = rnp.msgify(Image, depth_image_msg_data, encoding="16UC1")
            self.debug_depth_publisher.publish(depth_image_msg)
        if self.debug_pointcloud_publisher is not None:
            pointcloud_msg = self.create_pointcloud_msg(
                obs[self.depth_image_slice].reshape(*self.depth_image_final_resolution) * self.depth_image_clip_range[1]
                + self.depth_image_clip_range[0]
            )
            self.debug_pointcloud_publisher.publish(pointcloud_msg)
        return action, done

    def create_pointcloud_msg(self, depth_data: np.ndarray) -> PointCloud2:
        height, width = depth_data.shape
        vfov_deg = 58.0
        vfov_rad = np.deg2rad(vfov_deg)
        f = (height / 2.0) / np.tan(vfov_rad / 2.0)  # focal length in pixels (assuming fy = f)

        # Assuming square pixels, fx = fy = f, cx = width/2, cy = height/2
        cx = width / 2.0
        cy = height / 2.0

        # Create grid of pixel coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        depth = depth_data.astype(np.float32)

        x = (u - cx) * depth / f
        y = (v - cy) * depth / f
        z = depth

        # Stack to (H*W, 3)
        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

        # Filter invalid points (depth == 0)
        valid = (z > 0).flatten()
        points = points[valid]

        # Apply 90-degree rotation around +Y axis: new_x = z, new_y = y, new_z = -x
        rotated_points = np.empty_like(points)
        rotated_points[:, 0] = points[:, 2]  # new_x = old_z
        rotated_points[:, 1] = points[:, 1]  # new_y = old_y
        rotated_points[:, 2] = -points[:, 0]  # new_z = -old_x

        # Apply additional -90 degree rotation around +X axis: final_x = rotated_x, final_y = rotated_z, final_z = -rotated_y
        final_points = np.empty_like(rotated_points)
        final_points[:, 0] = rotated_points[:, 0]
        final_points[:, 1] = rotated_points[:, 2]
        final_points[:, 2] = -rotated_points[:, 1]

        # Create PointCloud2
        msg = PointCloud2()
        msg.header.stamp = self.ros_node.get_clock().now().to_msg()
        msg.header.frame_id = "d435_depth_link"
        msg.height = 1
        msg.width = len(final_points)
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12  # 3 floats * 4 bytes
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = final_points.astype(np.float32).tobytes()

        return msg

    """
    Agent specific observation functions for PerceptiveTrackerAgent.
    """

    def _get_visualizable_image_obs(self):
        """Return the depth image."""
        depth_image: np.ndarray = self.ros_node.rs_depth_data
        # normalize based on given range
        depth_image = np.clip(depth_image, self.depth_image_clip_range[0], self.depth_image_clip_range[1])
        if self.depth_image_shall_normalize:
            depth_image = (depth_image - self.depth_image_clip_range[0]) / (
                self.depth_image_clip_range[1] - self.depth_image_clip_range[0]
            )
            if self.depth_image_normalized_output_range is not None:
                depth_image = (
                    depth_image
                    * (self.depth_image_normalized_output_range[1] - self.depth_image_normalized_output_range[0])
                    + self.depth_image_normalized_output_range[0]
                )
        # gaussian blur the depth image
        if self.depth_image_gaussian_blur_kernel_size is not None:
            depth_image = cv2.GaussianBlur(
                depth_image,
                (self.depth_image_gaussian_blur_kernel_size, self.depth_image_gaussian_blur_kernel_size),
                self.depth_image_gaussian_blur_sigma,
            )
        # crop the depth image
        depth_image = depth_image[
            self.depth_image_crop_region[0] : -self.depth_image_crop_region[1],
            self.depth_image_crop_region[2] : -self.depth_image_crop_region[3],
        ]
        # resize the depth image to the final resolution
        depth_image = cv2.resize(
            depth_image,
            (self.depth_image_final_resolution[1], self.depth_image_final_resolution[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        return depth_image
