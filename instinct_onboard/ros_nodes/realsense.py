import multiprocessing as mp
import time
from typing import Tuple

import cv2
import numpy as np
import pyrealsense2 as rs

from .ros_real import Ros2Real


class RealSenseCamera:
    def __init__(self, resolution: Tuple[int, int], fps: int):
        self.resolution = resolution  # (width, height)
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(
            rs.stream.depth,
            self.resolution[0],
            self.resolution[1],
            rs.format.z16,
            fps,
        )
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.depth)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

        # get frame with longer waiting time to start the system
        # I know what's going on, but when enabling rgb, this solves the problem.
        _ = self.pipeline.wait_for_frames(1000)  # 1000 ms

        self.build_opencv_filters()

    def get_frame(self) -> rs.depth_frame or None:
        # read from pyrealsense2, preprocess and write the model embedding to the buffer
        timeout_ms = int(1000 / self.fps)  # ms
        frames = self.pipeline.wait_for_frames(timeout_ms * 2)
        depth_frame = frames.get_depth_frame()
        return depth_frame

    def get_camera_data(self) -> np.ndarray or None:
        depth_frame = self.get_frame()
        if depth_frame is None:
            return None
        depth_data = self.apply_opencv_filters(depth_frame)
        return depth_data  # (height, width)

    def build_opencv_filters(self):
        """Build the OpenCV filters."""
        assert cv2.cuda.getCudaEnabledDeviceCount() > 0, "CUDA must be available for OpenCV to use GPU acceleration."

    def apply_opencv_filters(self, depth_frame: rs.depth_frame) -> np.ndarray:
        """Apply the OpenCV filters to the depth image."""
        depth_np = np.asanyarray(depth_frame.get_data(), dtype=np.float32) * self.depth_scale
        to_inpaint_mask = (depth_np < 0.2).astype(np.uint8)
        depth_np = cv2.inpaint(depth_np, to_inpaint_mask, 3, cv2.INPAINT_NS)
        return depth_np


def camera_process_func(resolution, fps, request_queue, result_queue):
    camera = RealSenseCamera(resolution, fps)
    while True:
        depth_data = camera.get_camera_data()
        result_queue.put(depth_data)
        try:
            signal = request_queue.get_nowait()
            if signal is None:
                break
        except mp.queues.Empty:
            pass


class RsCameraNodeMixin:
    """
    Mixin for camera sensor or processing nodes.
    Extend this class when implementing a ROS2 node related to camera sensing or image streams.
    """

    def __init__(
        self,
        *args,
        rs_resolution: Tuple[int, int] = (480, 270),  # (width, height)
        rs_fps: int = 60,
        camera_individual_process: bool = False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        # Add any depth-specific initialization here
        self.rs_resolution = rs_resolution
        self.rs_fps = rs_fps
        self.camera_individual_process = camera_individual_process
        self.camera = None
        self.camera_process = None
        self.request_queue = None
        self.result_queue = None

    def initialize_camera(self):
        """Initialize the RealSense camera with the specified configuration."""
        if self.camera_individual_process:
            self.request_queue = mp.Queue()
            self.result_queue = mp.Queue()
            self.camera_process = mp.Process(
                target=camera_process_func,
                args=(self.rs_resolution, self.rs_fps, self.request_queue, self.result_queue),
            )
            self.camera_process.start()
            # We don't set self.camera, as it's in another process
            # Get depth_scale by requesting a frame or separately
            temp_depth = (
                self.get_rs_data()
            )  # Dummy call to get scale, but actually scale is not fetched; need to adjust
            self.rs_depth_scale = None  # TODO: Fetch depth_scale from process if needed
        else:
            self.camera = RealSenseCamera(
                resolution=self.rs_resolution,
                fps=self.rs_fps,
            )
            self.rs_depth_scale = self.camera.depth_scale

    def get_rs_data(self) -> np.ndarray or None:
        if self.camera_individual_process:
            if self.camera_process is None:
                raise ValueError("Camera not initialized. Call initialize_camera first.")
            # Dump queue and get latest
            latest = self.result_queue.get()  # Block until at least one
            while not self.result_queue.empty():
                try:
                    latest = self.result_queue.get_nowait()
                except mp.queues.Empty:
                    break
            return latest
        else:
            if self.camera is None:
                raise ValueError("Camera not initialized. Call initialize_camera first.")
            return self.camera.get_camera_data()  # (height, width)

    def destroy_node(self):
        if self.camera_individual_process and self.camera_process:
            self.request_queue.put(None)  # Signal to exit
            self.camera_process.join(timeout=1.0)
            if self.camera_process.is_alive():
                print("Warning: Camera process did not terminate gracefully, forcing terminate.")
                self.camera_process.terminate()
                self.camera_process.join()
        super().destroy_node()


class RsCameraNode(RsCameraNodeMixin, Ros2Real):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize_camera()
