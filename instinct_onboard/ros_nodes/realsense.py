from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from .unitree import UnitreeNode

REALSENSE_CALLBACK_FREQUENCY_CHECK_INTERVAL = 500


class RealSenseCamera:
    def __init__(self, resolution: tuple[int, int], fps: int):
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

    def get_frame(self) -> rs.depth_frame or None:
        # read from pyrealsense2, preprocess and write the model embedding to the buffer
        timeout_ms = int(1000 / self.fps)  # ms
        frames = self.pipeline.wait_for_frames(timeout_ms * 2)
        depth_frame = frames.get_depth_frame()
        return depth_frame

    def get_camera_data(self) -> np.array or None:
        depth_frame = self.get_frame()
        if depth_frame is None:
            return None
        # Apply Realsense Filters only if needed. Do not apply any OpenCV filters here.
        # Leave to each of the agents to apply the filters, because it may be different for each agent.
        depth_data = np.asanyarray(depth_frame.get_data(), dtype=np.float32) * self.depth_scale
        return depth_data


def camera_process_func(
    resolution: tuple[int, int],
    fps: int,
    request_queue: mp.Queue,
    result_queue: mp.Queue,
    camera_process_affinity: set[int] | None,
) -> None:
    if camera_process_affinity is not None:
        os.sched_setaffinity(os.getpid(), camera_process_affinity)
    camera = RealSenseCamera(resolution, fps)
    while True:
        depth_data = camera.get_camera_data()
        result_queue.put((time.time(), np.copy(depth_data)))
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
        rs_resolution: tuple[int, int] = (480, 270),  # (width, height)
        rs_fps: int = 60,
        camera_individual_process: bool = False,
        main_process_affinity: set[int] | None = None,
        camera_process_affinity: set[int] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Add any depth-specific initialization here
        self.rs_resolution = rs_resolution
        self.rs_fps = rs_fps
        self.camera_individual_process = camera_individual_process
        self.main_process_affinity = main_process_affinity
        self.camera_process_affinity = camera_process_affinity
        self.camera = None
        self.camera_process = None
        self.request_queue = None
        self.result_queue = None
        self.initialize_camera()

    def initialize_camera(self):
        """Initialize the RealSense camera with the specified configuration."""
        if self.camera_individual_process:
            self.request_queue = mp.Queue()
            self.result_queue = mp.Queue()
            self.camera_process = mp.Process(
                target=camera_process_func,
                args=(
                    self.rs_resolution,
                    self.rs_fps,
                    self.request_queue,
                    self.result_queue,
                    self.camera_process_affinity,
                ),
            )
            self.camera_process.start()
            if self.main_process_affinity is not None:
                os.sched_setaffinity(os.getpid(), self.main_process_affinity)
            # We don't set self.camera, as it's in another process
            # Get depth_scale by requesting a frame or separately
            self.refresh_rs_data()  # Dummy call to refresh the depth data, but actually scale is not fetched; need to adjust
        else:
            self.camera = RealSenseCamera(
                resolution=self.rs_resolution,
                fps=self.rs_fps,
            )

    def start_ros_handlers(self):
        self.realsense_timer = self.create_timer(1.0 / self.rs_fps, self.realsense_callback)
        if REALSENSE_CALLBACK_FREQUENCY_CHECK_INTERVAL > 1:
            self.realsense_timer_counter = 0
            self.realsense_timer_counter_time = time.time()
            self.realsense_callback_time_consumptions = queue.Queue(maxsize=REALSENSE_CALLBACK_FREQUENCY_CHECK_INTERVAL)
        super().start_ros_handlers()

    def realsense_callback(self):
        """Callback to ensure the depth data is always updated.
        This is used to ensure the depth data is always updated, even if the camera is not publishing.
        """
        realsense_callback_start_time = time.time()
        refreshed = self.refresh_rs_data()
        if REALSENSE_CALLBACK_FREQUENCY_CHECK_INTERVAL > 1:
            self.realsense_callback_time_consumptions.put(time.time() - realsense_callback_start_time)
            if refreshed:
                self.realsense_timer_counter += 1
            if self.realsense_timer_counter % REALSENSE_CALLBACK_FREQUENCY_CHECK_INTERVAL == 0:
                time_consumptions = [
                    self.realsense_callback_time_consumptions.get()
                    for _ in range(REALSENSE_CALLBACK_FREQUENCY_CHECK_INTERVAL)
                ]
                self.get_logger().info(
                    f"Actual realsense refreshed frequency: {(REALSENSE_CALLBACK_FREQUENCY_CHECK_INTERVAL / (time.time() - self.realsense_timer_counter_time)):.2f} Hz. Mean time consumption: {np.mean(time_consumptions):.4f} s."
                )
                self.realsense_timer_counter = 0
                self.realsense_timer_counter_time = time.time()

    def refresh_rs_data(self) -> bool:
        """Currently refresh the depth data only."""
        refreshed = False
        if self.camera_individual_process:
            if self.camera_process is None:
                raise ValueError("Camera not initialized. Call initialize_camera first.")
            # Dump queue and get latest
            if not hasattr(self, "rs_depth_data"):
                rs_timestamp, self.rs_depth_data = self.result_queue.get()  # Block and wait for the first depth data
                refreshed = True
            while self.result_queue.qsize() > 0:
                try:
                    rs_timestamp, self.rs_depth_data = self.result_queue.get(timeout=0.01)
                    refreshed = True
                except mp.queues.Empty:
                    print("Realsense depth data queue is empty. Breaking.")
                    break
            self.get_logger().info(
                f"Realsense depth data delayed: {(time.time() - rs_timestamp):.4f} s. queue size: {self.result_queue.qsize()}",
                throttle_duration_sec=2.0,
            )
        else:
            if self.camera is None:
                raise ValueError("Camera not initialized. Call initialize_camera first.")
            self.rs_depth_data = self.camera.get_camera_data()  # (height, width)
            refreshed = True
        return refreshed

    def destroy_node(self):
        if self.camera_individual_process and self.camera_process:
            self.request_queue.put(None)  # Signal to exit
            self.camera_process.join(timeout=1.0)
            if self.camera_process.is_alive():
                print("Warning: Camera process did not terminate gracefully, forcing terminate.")
                self.camera_process.terminate()
                self.camera_process.join()
        super().destroy_node()


class UnitreeRsCameraNode(RsCameraNodeMixin, UnitreeNode):
    pass
