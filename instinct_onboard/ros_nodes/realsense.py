from __future__ import annotations

import ctypes
import multiprocessing as mp
import multiprocessing.shared_memory as mp_shm
import os
import queue
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from rclpy.callback_groups import ReentrantCallbackGroup

from .unitree import UnitreeNode

REALSENSE_PROCESS_FREQUENCY_CHECK_INTERVAL = 500


class MpSharedHeader(ctypes.Structure):
    _fields_ = [
        ("timestamp", ctypes.c_double),  # bytes: 8
        ("writer_status", ctypes.c_uint32),  # bytes: 4, 0: idle, 1: writing
        ("writer_termination_signal", ctypes.c_uint32),  # bytes: 4, 0: alive, 1: should terminate
        ("_pad", ctypes.c_uint32 * 4),  # bytes: 16, pad to 32 bytes
    ]


SIZE_OF_MP_SHARED_HEADER = ctypes.sizeof(MpSharedHeader)  # bytes: 32
assert SIZE_OF_MP_SHARED_HEADER == 32


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
    shm_name: str,
    camera_process_affinity: set[int] | None,
) -> None:
    if camera_process_affinity is not None:
        os.sched_setaffinity(os.getpid(), camera_process_affinity)
    camera = RealSenseCamera(resolution, fps)
    shared_memory = mp.shared_memory.SharedMemory(name=shm_name)
    header = MpSharedHeader.from_buffer(shared_memory.buf)
    image_buffer = np.ndarray(
        resolution[::-1], dtype=np.float32, buffer=shared_memory.buf, offset=SIZE_OF_MP_SHARED_HEADER
    )
    camera_process_start_time = time.time()
    camera_process_counter = 0
    while True:
        camera_data = camera.get_camera_data()
        # mark in header to start writing
        header.writer_status = 1
        # write the camera data to the shared memory
        image_buffer[:] = camera_data
        header.timestamp = time.time()
        # mark in header to stop writing
        header.writer_status = 0
        # check if the writer termination signal is set
        if header.writer_termination_signal == 1:
            print("Writer termination signal set, exiting camera process.")
            header = None
            image_buffer = None
            break
        camera_process_counter += 1
        if camera_process_counter % REALSENSE_PROCESS_FREQUENCY_CHECK_INTERVAL == 0:
            print(
                f"Realsense camera process running at {(camera_process_counter / (time.time() - camera_process_start_time)):.4f} Hz."
            )
            camera_process_counter = 0
            camera_process_start_time = time.time()
    shared_memory.close()  # unlink in the main process


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
            # self.rs_rgb_data = None # Todo: add rgb data support
            self.rs_depth_data = np.zeros(self.rs_resolution[::-1], dtype=np.float32)
            shm_size = (
                SIZE_OF_MP_SHARED_HEADER
                + np.prod(self.rs_resolution[::-1]) * np.dtype(self.rs_depth_data.dtype).itemsize
            )
            self.rs_shared_memory = mp_shm.SharedMemory(create=True, size=shm_size)
            self.rs_shared_header = MpSharedHeader.from_buffer(self.rs_shared_memory.buf)
            self.rs_image_buffer = np.ndarray(
                self.rs_resolution[::-1],
                dtype=np.float32,
                buffer=self.rs_shared_memory.buf,
                offset=SIZE_OF_MP_SHARED_HEADER,
            )
            self.rs_data_fresh_counter = 0
            self.camera_process = mp.Process(
                target=camera_process_func,
                args=(
                    self.rs_resolution,
                    self.rs_fps,
                    self.rs_shared_memory.name,
                    self.camera_process_affinity,
                ),
                daemon=True,
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

    def refresh_rs_data(self) -> bool:
        """Currently refresh the depth data only."""
        refreshed = False
        if self.camera_individual_process:
            if self.camera_process is None:
                raise ValueError("Camera not initialized. Call initialize_camera first.")
            # Check if process is alive
            if not self.camera_process.is_alive():
                raise ValueError("Camera process is not alive. Exiting.")
            # Dump queue and get latest
            if self.rs_shared_header.writer_status == 0:
                rs_timestamp = self.rs_shared_header.timestamp
                self.rs_depth_data[:] = self.rs_image_buffer
                self.get_logger().info(
                    f"Realsense depth data delayed: {(time.time() - rs_timestamp):.4f} s.", throttle_duration_sec=5.0
                )
                refreshed = True
            self.rs_data_fresh_counter += 1
        else:
            if self.camera is None:
                raise ValueError("Camera not initialized. Call initialize_camera first.")
            self.rs_depth_data = self.camera.get_camera_data()  # (height, width)
            refreshed = True
        return refreshed

    def destroy_node(self):
        if self.camera_individual_process and self.camera_process:
            self.rs_shared_header.writer_termination_signal = 1
            self.camera_process.join(timeout=1.0)
            if self.camera_process.is_alive():
                self.get_logger().warn("Camera process is still alive after timeout. Terminating and joining.")
                self.camera_process.terminate()
                self.camera_process.join()
            self.rs_image_buffer = None
            self.rs_shared_header = None
            self.camera_process = None
            self.rs_shared_memory.close()
            self.rs_shared_memory.unlink()
            self.rs_shared_memory = None
        super().destroy_node()


class UnitreeRsCameraNode(RsCameraNodeMixin, UnitreeNode):
    pass
