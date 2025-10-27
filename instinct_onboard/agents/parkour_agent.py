from __future__ import annotations

import multiprocessing as mp
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
import ros2_numpy as rnp
import yaml
from sensor_msgs.msg import CameraInfo, Image

from instinct_onboard.agents.base import OnboardAgent
from instinct_onboard.ros_nodes.ros_real import Ros2Real
from instinct_onboard.utils import CircularBuffer


def depth_worker(queue, config):
    rs_pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(
        rs.stream.depth,
        config["rs_resolution"][0],
        config["rs_resolution"][1],
        rs.format.z16,
        config["rs_frequency"],
    )
    rs_profile = rs_pipeline.start(rs_config)
    depth_scale = rs_profile.get_device().first_depth_sensor().get_depth_scale()
    depth_image_buffer = CircularBuffer(length=config["rs_frequency"])

    while True:
        # print("Current time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        # read from pyrealsense2, preprocess and write the model latent to the buffer
        rs_frame = rs_pipeline.wait_for_frames()  # ms
        # frame_timestamp = rs_frame.get_timestamp()
        depth_frame = rs_frame.get_depth_frame()
        # current_time_ms = int(time.time() * 1000)
        # print("RealSense frame retrieval time: {:.4f} ms, current time: {} ms.".format(frame_timestamp, current_time_ms))
        if not depth_frame:
            raise RuntimeError("No depth frame")
        # start_get = time.time()
        depth_image_np = np.asanyarray(depth_frame.get_data(), dtype=np.float32) * depth_scale
        # depth_input_msg = rnp.msgify(Image, np.asanyarray(depth_image_np/self.depth_scale, dtype=np.uint16), encoding="16UC1")

        depth_image = cv2.resize(depth_image_np, config["output_resolution"], interpolation=cv2.INTER_NEAREST)

        if "crop_region" in config:
            shape = depth_image.shape
            x1, x2, y1, y2 = config["crop_region"]
            depth_image = depth_image[x1 : shape[0] - x2, y1 : shape[1] - y2]

        mask = (depth_image < 0.2).astype(np.uint8)
        depth_image = cv2.inpaint(depth_image, mask, 3, cv2.INPAINT_NS)

        if "blind_spot_crop" in config:
            shape = depth_image.shape
            x1, x2, y1, y2 = config["blind_spot_crop"]
            depth_image[:x1, :] = 0
            depth_image[shape[0] - x2 :, :] = 0
            depth_image[:, :y1] = 0
            depth_image[:, shape[1] - y2 :] = 0
        if "gaussian_kernel_size" in config:
            depth_image = cv2.GaussianBlur(
                depth_image, config["gaussian_kernel_size"], config["gaussian_sigma"], config["gaussian_sigma"]
            )

        filt_m = np.clip(depth_image, config["depth_range"][0], config["depth_range"][1])
        filt_norm = (filt_m - config["depth_range"][0]) / (config["depth_range"][1] - config["depth_range"][0])

        output_norm = (
            filt_norm * (config["depth_output_range"][1] - config["depth_output_range"][0])
            + config["depth_output_range"][0]
        )
        # depth_image = depth_image_np[self.y_valid, self.x_valid]
        # publish the depth image input to ros topic
        depth_image_buffer.append(output_norm)
        if queue.full():
            try:
                queue.get_nowait()
            except:
                pass
        queue.put(depth_image_buffer)
        # print("Depth processing time: {:.4f} ms.".format((time.time() - start_get) * 1000))


class ParkourAgent(OnboardAgent):
    rs_resolution = (480, 270)
    rs_frequency = 60
    visualize_depth = False
    publish_depth = True

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
        self.cmd_px_range = [0.0, 0.8]
        self.cmd_nx_range = [0.0, 0.0]
        self.cmd_py_range = [0.0, 0.0]
        self.cmd_ny_range = [0.0, 0.0]
        self.cmd_pyaw_range = [0.0, 1.0]
        self.cmd_nyaw_range = [0.0, 1.0]
        self.move_by_wireless_remote = True
        self._parse_obs_config()
        self._parse_action_config()
        self._parse_depth_image_config()
        self._load_models()
        self._start_rs_pipeline()
        # self._depth_thread = threading.Thread(target=self.get_depth_frame, daemon=True)
        # self._depth_thread.start()

        self.depth_input_pub = self.ros_node.create_publisher(
            Image,
            "/depth_image",
            1,
        )

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
        self.depth_configs = {}
        self.output_resolution = [
            self.cfg["scene"]["camera"]["pattern_cfg"]["width"],
            self.cfg["scene"]["camera"]["pattern_cfg"]["height"],
        ]
        self.depth_configs["output_resolution"] = self.output_resolution

        self.depth_range = self.cfg["scene"]["camera"]["noise_pipeline"]["depth_normalization"]["depth_range"]
        self.depth_configs["depth_range"] = self.depth_range

        if self.cfg["scene"]["camera"]["noise_pipeline"]["depth_normalization"]["normalize"]:
            self.depth_output_range = self.cfg["scene"]["camera"]["noise_pipeline"]["depth_normalization"][
                "output_range"
            ]
        else:
            self.depth_output_range = self.depth_range
        self.depth_configs["depth_output_range"] = self.depth_output_range

        if "crop_and_resize" in self.cfg["scene"]["camera"]["noise_pipeline"]:
            self.crop_region = self.cfg["scene"]["camera"]["noise_pipeline"]["crop_and_resize"]["crop_region"]
            self.depth_configs["crop_region"] = self.crop_region
        if "gaussian_blur" in self.cfg["scene"]["camera"]["noise_pipeline"]:
            self.gaussian_kernel_size = (
                self.cfg["scene"]["camera"]["noise_pipeline"]["gaussian_blur"]["kernel_size"],
                self.cfg["scene"]["camera"]["noise_pipeline"]["gaussian_blur"]["kernel_size"],
            )
            self.gaussian_sigma = self.cfg["scene"]["camera"]["noise_pipeline"]["gaussian_blur"]["sigma"]
            self.depth_configs["gaussian_kernel_size"] = self.gaussian_kernel_size
            self.depth_configs["gaussian_sigma"] = self.gaussian_sigma
        if "blind_spot" in self.cfg["scene"]["camera"]["noise_pipeline"]:
            self.blind_spot_crop = self.cfg["scene"]["camera"]["noise_pipeline"]["blind_spot"]["crop_region"]
            self.depth_configs["blind_spot_crop"] = self.blind_spot_crop
        self.depth_width = (
            self.output_resolution[0] - self.crop_region[2] - self.crop_region[3]
            if hasattr(self, "crop_region")
            else self.output_resolution[0]
        )
        self.depth_height = (
            self.output_resolution[1] - self.crop_region[0] - self.crop_region[1]
            if hasattr(self, "crop_region")
            else self.output_resolution[1]
        )
        self.depth_configs["depth_width"] = self.depth_width
        self.depth_configs["depth_height"] = self.depth_height
        self.depth_configs["rs_frequency"] = self.rs_frequency
        self.depth_configs["rs_resolution"] = self.rs_resolution
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

    def _parse_observation_function(self, obs_name, obs_config):
        obs_func = obs_config["func"].split(":")[-1]  # get the function name from the config
        if obs_func == "depth_image":
            obs_name = "depth_latent"
            if hasattr(self, f"_get_{obs_name}_obs"):
                self.obs_funcs[obs_name] = getattr(self, f"_get_{obs_name}_obs")
                return
            else:
                raise ValueError(f"Unknown observation function for observation {obs_name}")
        return super()._parse_observation_function(obs_name, obs_config)

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
        self.depth_queue = mp.Queue(maxsize=1)
        self.depth_proc = mp.Process(target=depth_worker, args=(self.depth_queue, self.depth_configs), daemon=True)
        self.depth_proc.start()

    # def get_depth_frame(self):
    #     while True:
    #         # print("Current time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    #         # read from pyrealsense2, preprocess and write the model latent to the buffer
    #         rs_frame = self.rs_pipeline.wait_for_frames()  # ms
    #         # frame_timestamp = rs_frame.get_timestamp()
    #         depth_frame = rs_frame.get_depth_frame()
    #         # current_time_ms = int(time.time() * 1000)
    #         # print("RealSense frame retrieval time: {:.4f} ms, current time: {} ms.".format(frame_timestamp, current_time_ms))
    #         if not depth_frame:
    #             self.ros_node.get_logger().error("No depth frame", throttle_duration_sec=1)
    #             return
    #         # apply relsense filters
    #         # start_filter = time.time()
    #         # for rs_filter in self.rs_filters:
    #         #     depth_frame = rs_filter.process(depth_frame)
    #         # filter_time = time.time() - start_filter
    #         # start_get = time.time()
    #         depth_image_np = np.asanyarray(depth_frame.get_data(), dtype=np.float32) * self.depth_scale
    #         # depth_input_msg = rnp.msgify(Image, np.asanyarray(depth_image_np/self.depth_scale, dtype=np.uint16), encoding="16UC1")

    #         depth_image = cv2.resize(depth_image_np, self.output_resolution, interpolation=cv2.INTER_NEAREST)
    #         # depth_image = depth_image_np[self.y_valid, self.x_valid]

    #         if self.visualize_depth:
    #             # display normalized depth as 8-bit for visualization
    #             vis = (depth_image/2.5 * 255).astype(np.uint8)
    #             filename = f"depth.png"
    #             cv2.imwrite(filename, vis)
    #             # cv2.imshow("Depth Image", vis)

    #         if hasattr(self, "crop_region"):
    #             shape = depth_image.shape
    #             x1, x2, y1, y2 = self.crop_region
    #             depth_image = depth_image[x1 : shape[0] - x2, y1 : shape[1] - y2]

    #         mask = (depth_image < 0.2).astype(np.uint8)
    #         depth_image = cv2.inpaint(depth_image, mask, 3, cv2.INPAINT_NS)

    #         if hasattr(self, "blind_spot_crop"):
    #             shape = depth_image.shape
    #             x1, x2, y1, y2 = self.blind_spot_crop
    #             depth_image[:x1, :] = 0
    #             depth_image[shape[0] - x2 :, :] = 0
    #             depth_image[:, :y1] = 0
    #             depth_image[:, shape[1] - y2 :] = 0
    #         if hasattr(self, "gaussian_kernel_size"):
    #             depth_image = cv2.GaussianBlur(
    #                 depth_image, self.gaussian_kernel_size, self.gaussian_sigma, self.gaussian_sigma
    #             )

    #         if self.publish_depth:
    #             depth_input_msg = rnp.msgify(Image, np.asanyarray(depth_image/self.depth_scale, dtype=np.uint16), encoding="16UC1")
    #             depth_input_msg.header.stamp = self.ros_node.get_clock().now().to_msg()
    #             depth_input_msg.header.frame_id = "d435_depth_link"
    #             self.depth_input_pub.publish(depth_input_msg)

    #         filt_m = np.clip(depth_image, self.depth_range[0], self.depth_range[1])
    #         filt_norm = (filt_m - self.depth_range[0]) / (self.depth_range[1] - self.depth_range[0])

    #         output_norm = (
    #             filt_norm * (self.depth_output_range[1] - self.depth_output_range[0]) + self.depth_output_range[0]
    #         )

    #         # publish the depth image input to ros topic
    #         self.ros_node.get_logger().info("depth range: {}-{}".format(*self.depth_range), once=True)
    #         self.depth_image_buffer.append(output_norm)
    #         # print("Depth processing time: {:.4f} ms.".format((time.time() - start_get) * 1000))

    def reset(self):
        """Reset the agent state and the rosbag reader."""
        pass

    def step(self):
        # cur_time=time.time()
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
        if self.visualize_depth:
            self._vis_depth_obs(depth_obs.reshape(-1, self.depth_height, self.depth_width))

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
        # print(time.time()-cur_time)

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
        if not hasattr(self, "depth_queue"):
            return np.zeros((len(self.depth_obs_indices), self.depth_height, self.depth_width), dtype=np.float32)
        try:
            while True:
                depth_images = self.depth_queue.get_nowait().buffer
                self._last_depth_images = depth_images
        except:
            pass

        if self._last_depth_images is None:
            return np.zeros((len(self.depth_obs_indices), self.depth_height, self.depth_width), dtype=np.float32)
        return self._last_depth_images[self.depth_obs_indices, ...]

    def _vis_depth_obs(self, depth_obs: np.ndarray):
        depth_tiles = (np.clip(depth_obs[0], 0.0, 1.0) * 255).astype(np.uint8)
        rows, cols = 2, 4
        tile_h, tile_w = depth_tiles.shape[1], depth_tiles.shape[2]
        grid = np.zeros((rows * tile_h, cols * tile_w), dtype=np.uint8)
        for idx in range(depth_tiles.shape[0]):
            r, c = divmod(idx, cols)
            grid[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = depth_tiles[idx]
        cv2.imwrite("depth_obs_grid.png", grid)


class ParkourColdStartAgent(OnboardAgent):
    def __init__(
        self, logdir: str, dof_max_err: float, start_steps: int, ros_node: Ros2Real, joint_target_pos: np.array = None
    ):
        """Initialize the parkour cold start agent.
        Args:
            startup_step_size (float): The step size for the cold start agent to move the joints.
            ros_node (Ros2Real): The ROS node instance to interact with the robot.
        """
        super().__init__(logdir, ros_node)
        super()._parse_action_config()
        self.ros_node = ros_node
        self.dof_max_err = dof_max_err
        self.start_steps = start_steps
        self.joint_target_pos = np.zeros(self.ros_node.NUM_JOINTS) if joint_target_pos is None else joint_target_pos
        self._p_gains = np.ones(self.ros_node.NUM_JOINTS, dtype=np.float32) * 10.0  # default p_gains
        self._d_gains = np.zeros(self.ros_node.NUM_JOINTS, dtype=np.float32)  # default d_gains
        # self._p_gains = self._p_gains * 2  # default p_gains
        # self._d_gains = self._d_gains * 2  # default d_gains

    def step(self) -> tuple[np.ndarray, bool]:
        """Run a single step of the cold start agent. This will turn on the motors to a desired position.
        The desired position is defined in the robot configuration.
        """
        if self.step_count < self.start_steps:
            actions = self.start_dof_pos + self.dof_pos_err * (self.step_count + 1) / self.start_steps
            done = False
            self.step_count += 1
        else:
            dof_pos_err = self.joint_target_pos - self._get_joint_pos_rel_obs()
            err_large_mask = np.abs(dof_pos_err) > self.dof_max_err
            done = not err_large_mask.any()
            if not done:
                print(
                    f"Current ColdStartAgent gets max error {np.round(np.max(np.abs(dof_pos_err)), decimals=2)}",
                    end="\r",
                )
                raise SystemExit
            else:
                actions = np.zeros(self.ros_node.NUM_JOINTS)

        return actions, done

    def reset(self):
        """Reset the agent."""
        self.start_dof_pos = self._get_joint_pos_rel_obs()
        self.dof_pos_err = self.joint_target_pos - self.start_dof_pos
        self.step_count = 0


class ParkourStandAgent(ParkourAgent):
    def __init__(
        self,
        logdir: str,
        ros_node: Ros2Real,
    ):
        super().__init__(logdir, ros_node)

    def _start_rs_pipeline(self):
        pass
