from __future__ import annotations

import os
import re
import threading
import time
from typing import Tuple

import cv2
import numpy as np
import onnxruntime as ort
import prettytable
import pyrealsense2 as rs
import yaml

from instinct_onboard.agents.base import OnboardAgent
from instinct_onboard.ros_nodes.ros_real import Ros2Real
from instinct_onboard.utils import CircularBuffer


class ParkourDogAgent(OnboardAgent):
    rs_resolution = (480, 270)
    rs_frequency = 60
    visualize_depth = False

    def __init__(
        self,
        logdir: str,
        ros_node: Ros2Real,
    ):
        super().__init__(logdir, ros_node)
        assert cv2.cuda.getCudaEnabledDeviceCount() > 0, "ParkourAgent requires a CUDA-enabled OpenCV installation."
        self.ort_sessions = dict()
        self.lin_vel_deadband = 0.15
        self.ang_vel_deadband = 0.15
        self.cmd_px_range = [0.0, 1.0]
        self.cmd_nx_range = [0.0, 0.0]
        self.cmd_py_range = [0.0, 0.3]
        self.cmd_ny_range = [0.0, 0.3]
        self.cmd_pyaw_range = [0.0, 1.0]
        self.cmd_nyaw_range = [0.0, 1.0]
        self.move_by_wireless_remote = True
        self._parse_obs_config()
        self._parse_action_config()
        self._parse_depth_image_config()
        self._load_models()
        self._start_rs_pipeline()
        self._depth_thread = threading.Thread(target=self.get_depth_frame, daemon=True)
        self._depth_thread.start()

    def _parse_obs_config(self):
        super()._parse_obs_config()
        with open(os.path.join(self.logdir, "params", "agent.yaml")) as f:
            self.agent_cfg = yaml.unsafe_load(f)
        all_obs_names = list(self.obs_funcs.keys())
        self.proprio_obs_names = [obs_name for obs_name in all_obs_names if "depth" not in obs_name]
        print(f"ParkourAgent proprioception names: {self.proprio_obs_names}")
        self.depth_obs_names = [obs_name for obs_name in all_obs_names if "depth" in obs_name]
        assert len(self.depth_obs_names) == 1, "Only support one depth observation for now."
        print(f"ParkourAgent depth observation names: {self.depth_obs_names}")
        table = prettytable.PrettyTable()
        table.field_names = ["Observation Name", "Function"]
        for obs_name, func in self.obs_funcs.items():
            table.add_row([obs_name, func.__name__])
        print("Observation functions:")
        print(table)

    def _parse_action_config(self):
        super()._parse_action_config()
        self._zero_action_joints = np.zeros(self.ros_node.NUM_ACTIONS, dtype=np.float32)
        for action_names, action_config in self.cfg["actions"].items():
            for i in range(self.ros_node.NUM_JOINTS):
                name = self.ros_node.sim_joint_names[i]
                if "default_joint_names" in action_config:
                    for _, joint_name_expr in enumerate(action_config["default_joint_names"]):
                        if re.search(joint_name_expr, name):
                            self._zero_action_joints[i] = 1.0

    def _parse_depth_image_config(self):
        self.output_resolution = [
            self.cfg["scene"]["camera"]["pattern_cfg"]["width"],
            self.cfg["scene"]["camera"]["pattern_cfg"]["height"],
        ]
        self.depth_width = self.output_resolution[0]
        self.depth_height = self.output_resolution[1]
        self.depth_range = self.cfg["scene"]["camera"]["noise_pipeline"]["depth_normalization"]["depth_range"]
        if self.cfg["scene"]["camera"]["noise_pipeline"]["depth_normalization"]["normalize"]:
            self.depth_output_range = self.cfg["scene"]["camera"]["noise_pipeline"]["depth_normalization"][
                "output_range"
            ]
        else:
            self.depth_output_range = self.depth_range
        if "gaussian_blur" in self.cfg["scene"]["camera"]["noise_pipeline"]:
            self.gaussian_kernel_size = (
                self.cfg["scene"]["camera"]["noise_pipeline"]["gaussian_blur"]["kernel_size"],
                self.cfg["scene"]["camera"]["noise_pipeline"]["gaussian_blur"]["kernel_size"],
            )
            self.gaussian_sigma = self.cfg["scene"]["camera"]["noise_pipeline"]["gaussian_blur"]["sigma"]
        if "blind_spot" in self.cfg["scene"]["camera"]["noise_pipeline"]:
            self.blind_spot_crop = self.cfg["scene"]["camera"]["noise_pipeline"]["blind_spot"]["crop_region"]
        # For sample resize
        square_size = int(self.rs_resolution[0] // self.output_resolution[0])
        rows, cols = self.rs_resolution[1], self.rs_resolution[0]
        center_y_coords = np.arange(self.output_resolution[1]) * square_size + square_size // 2
        center_x_coords = np.arange(self.output_resolution[0]) * square_size + square_size // 2
        y_grid, x_grid = np.meshgrid(center_y_coords, center_x_coords, indexing="ij")
        valid_mask = (y_grid < rows) & (x_grid < cols)
        self.y_valid = np.clip(y_grid, 0, rows - 1)
        self.x_valid = np.clip(x_grid, 0, cols - 1)
        # For downsample history
        downsample_factor = self.cfg["observations"]["policy"]["depth_image"]["params"]["time_downsample_factor"]
        frames = int(
            (self.cfg["scene"]["camera"]["data_histories"]["distance_to_image_plane_noised"] - 1) / downsample_factor
            + 1
        )
        sim_frequency = int(1 / self.cfg["scene"]["camera"]["update_period"])
        real_downsample_factor = int(self.rs_frequency / sim_frequency * downsample_factor)
        self.depth_obs_indices = np.linspace(-1 - real_downsample_factor * (frames - 1), -1, frames).astype(int)
        print(f"Depth observation downsample indices: {self.depth_obs_indices}")
        self.depth_image_buffer = CircularBuffer(length=self.rs_frequency)

    def _load_models(self):
        """Load the ONNX model for the agent."""
        # load ONNX models
        ort_execution_providers = ort.get_available_providers()
        depth_encoder_path = os.path.join(self.logdir, "exported", "0-depth_encoder.onnx")
        self.ort_sessions["depth_encoder"] = ort.InferenceSession(depth_encoder_path, providers=ort_execution_providers)
        actor_path = os.path.join(self.logdir, "exported", "actor.onnx")
        self.ort_sessions["actor"] = ort.InferenceSession(actor_path, providers=ort_execution_providers)
        print(f"Loaded ONNX models from {self.logdir}")

    def _start_rs_pipeline(self):
        """Start the RealSense camera pipeline."""
        self.rs_pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(
            rs.stream.depth,
            self.rs_resolution[0],
            self.rs_resolution[1],
            rs.format.z16,
            self.rs_frequency,
        )
        self.rs_profile = self.rs_pipeline.start(rs_config)
        rs_depth_to_disparity_filter = rs.disparity_transform(True)
        rs_hole_filling_filter = rs.hole_filling_filter(
            1
        )  # 0: fill from left; 1: farthest from around; 2: nearest from around
        rs_spatial_filter = rs.spatial_filter()
        rs_spatial_filter.set_option(rs.option.filter_magnitude, 5)
        rs_spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.75)
        rs_spatial_filter.set_option(rs.option.filter_smooth_delta, 1)
        rs_spatial_filter.set_option(rs.option.holes_fill, 4)
        rs_temporal_filter = rs.temporal_filter()
        rs_temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.6)
        rs_temporal_filter.set_option(rs.option.filter_smooth_delta, 20)
        rs_diaparity_to_depth_filter = rs.disparity_transform(False)
        # using a list of filters to define the filtering order
        self.rs_filters = [
            # rs_decimation_filter,
            rs_depth_to_disparity_filter,
            rs_hole_filling_filter,
            rs_spatial_filter,
            rs_temporal_filter,
            rs_diaparity_to_depth_filter,
        ]
        self.depth_scale = self.rs_profile.get_device().first_depth_sensor().get_depth_scale()

    def get_depth_frame(self):
        while True:
            # print("Current time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            # read from pyrealsense2, preprocess and write the model latent to the buffer
            rs_frame = self.rs_pipeline.wait_for_frames()  # ms
            # frame_timestamp = rs_frame.get_timestamp()
            depth_frame = rs_frame.get_depth_frame()
            # current_time_ms = int(time.time() * 1000)
            # print("RealSense frame retrieval time: {:.4f} ms, current time: {} ms.".format(frame_timestamp, current_time_ms))
            if not depth_frame:
                self.ros_node.get_logger().error("No depth frame", throttle_duration_sec=1)
                return
            # apply relsense filters
            # start_filter = time.time()
            # for rs_filter in self.rs_filters:
            #     depth_frame = rs_filter.process(depth_frame)
            # filter_time = time.time() - start_filter
            # start_get = time.time()
            depth_image_np = np.asanyarray(depth_frame.get_data(), dtype=np.float32) * self.depth_scale
            depth_image = depth_image_np[self.y_valid, self.x_valid]

            mask = (depth_image < 0.05).astype(np.uint8)
            depth_image = cv2.inpaint(depth_image, mask, 3, cv2.INPAINT_NS)

            if hasattr(self, "blind_spot_crop"):
                shape = depth_image.shape
                x1, x2, y1, y2 = self.blind_spot_crop
                depth_image[:x1, :] = 0
                depth_image[shape[0] - x2 :, :] = 0
                depth_image[:, :y1] = 0
                depth_image[:, shape[1] - y2 :] = 0
            if hasattr(self, "gaussian_kernel_size"):
                depth_image = cv2.GaussianBlur(
                    depth_image, self.gaussian_kernel_size, self.gaussian_sigma, self.gaussian_sigma
                )

            filt_m = np.clip(depth_image, self.depth_range[0], self.depth_range[1])
            filt_norm = (filt_m - self.depth_range[0]) / (self.depth_range[1] - self.depth_range[0])

            if self.visualize_depth:
                # display normalized depth as 8-bit for visualization
                vis = (filt_norm * 255).astype(np.uint8)
                filename = f"depth.png"
                cv2.imwrite(filename, vis)

            output_norm = (
                filt_norm * (self.depth_output_range[1] - self.depth_output_range[0]) + self.depth_output_range[0]
            )

            # publish the depth image input to ros topic
            self.ros_node.get_logger().info("depth range: {}-{}".format(*self.depth_range), once=True)
            self.depth_image_buffer.append(output_norm)
            # print("Depth processing time: {:.4f} ms.".format((time.time() - start_get) * 1000))

    def reset(self):
        """Reset the agent state and the rosbag reader."""
        pass

    def step(self):
        """Perform a single step of the agent."""
        # pack actor MLP input
        proprio_obs = []
        for proprio_obs_name in self.proprio_obs_names:
            obs_term_value = self._get_single_obs_term(proprio_obs_name)
            proprio_obs.append(np.reshape(obs_term_value, (1, -1)).astype(np.float32))
        proprio_obs = np.concatenate(proprio_obs, axis=-1)

        depth_obs = (
            self._get_single_obs_term(self.depth_obs_names[0])
            .reshape(1, -1, self.depth_height, self.depth_width)
            .astype(np.float32)
        )
        # depth_obs*=0.
        depth_image_output = self.ort_sessions["depth_encoder"].run(
            None, {self.ort_sessions["depth_encoder"].get_inputs()[0].name: depth_obs}
        )[0]
        # run actor MLP
        actor_input = np.concatenate([proprio_obs, depth_image_output], axis=1)
        actor_input_name = self.ort_sessions["actor"].get_inputs()[0].name
        action = self.ort_sessions["actor"].run(None, {actor_input_name: actor_input})[0]
        action = action.reshape(-1)
        # reconstruct full action including zeroed joints
        mask = (self._zero_action_joints == 0).astype(bool)
        full_action = np.zeros(self.ros_node.NUM_ACTIONS, dtype=np.float32)
        full_action[mask] = action

        done = False

        return full_action, done

    """
    Agent specific observation functions for Parkour Agent.
    """

    def _get_base_lin_vel_zero_obs(self):
        """Return shape: (3,)"""
        return np.zeros((3,))

    def _get_base_velocity_obs(self):
        """Return shape: (3,)"""
        if self.move_by_wireless_remote:
            joy_stick_command = self.ros_node.joy_stick_command  # [Lx, Ly, Rx, Ry]
            # left-y for forward/backward
            ly = joy_stick_command[1]
            if ly > self.lin_vel_deadband:
                vx = (ly - self.lin_vel_deadband) / (1 - self.lin_vel_deadband)  # (0, 1)
                vx = vx * (self.cmd_px_range[1] - self.cmd_px_range[0]) + self.cmd_px_range[0]
            elif ly < -self.lin_vel_deadband:
                vx = (ly + self.lin_vel_deadband) / (1 - self.lin_vel_deadband)  # (-1, 0)
                vx = vx * (self.cmd_nx_range[1] - self.cmd_nx_range[0]) - self.cmd_nx_range[0]
            else:
                vx = 0
            # left-x for side moving left/right
            lx = -joy_stick_command[0]
            if lx > self.lin_vel_deadband:
                vy = (lx - self.lin_vel_deadband) / (1 - self.lin_vel_deadband)
                vy = vy * (self.cmd_py_range[1] - self.cmd_py_range[0]) + self.cmd_py_range[0]
            elif lx < -self.lin_vel_deadband:
                vy = (lx + self.lin_vel_deadband) / (1 - self.lin_vel_deadband)
                vy = vy * (self.cmd_ny_range[1] - self.cmd_ny_range[0]) - self.cmd_ny_range[0]
            else:
                vy = 0
            # right-x for turning left/right
            rx = -joy_stick_command[2]
            if rx > self.ang_vel_deadband:
                yaw = (rx - self.ang_vel_deadband) / (1 - self.ang_vel_deadband)
                yaw = yaw * (self.cmd_pyaw_range[1] - self.cmd_pyaw_range[0]) + self.cmd_pyaw_range[0]
            elif rx < -self.ang_vel_deadband:
                yaw = (rx + self.ang_vel_deadband) / (1 - self.ang_vel_deadband)
                yaw = yaw * (self.cmd_nyaw_range[1] - self.cmd_nyaw_range[0]) - self.cmd_nyaw_range[0]
            else:
                yaw = 0

            self.xyyaw_command = np.array([vx, vy, yaw], dtype=np.float32)
            return self.xyyaw_command

    def _get_joint_vel_rel_obs(self):
        """Return shape: (num_joints,)"""
        return self.ros_node.joint_vel_

    def _get_depth_latent_obs(self):
        """Return shape: (depth_latent_dim,)"""
        return self.ros_node.depth_latent_buffer

    def _get_last_action_obs(self):
        """Return shape: (num_active_joints,)"""
        actions = np.asarray(self.ros_node.actions).astype(np.float32)
        mask = (1.0 - self._zero_action_joints).astype(bool)
        return actions[mask]

    def _get_depth_image_downsample_obs(self):
        """Return shape: (num_active_joints,)"""
        depth_images = self.depth_image_buffer.buffer
        return depth_images[self.depth_obs_indices, ...]
