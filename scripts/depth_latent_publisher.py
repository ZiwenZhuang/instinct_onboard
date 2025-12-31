import os

import cv2
import numpy as np
import onnxruntime as ort
import pyrealsense2 as rs
import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class DepthLatentPublisher(Node):
    @staticmethod
    def add_arguments(parser):
        """Add command line arguments for the node."""
        parser.add_argument(
            "--publish_frequency",
            type=float,
            default=10.0,
            help="Frequency at which to publish depth latent messages.",
        )
        parser.add_argument(
            "--rs_resolution",
            type=tuple,
            default=(480, 270),
            help="Resolution for the RealSense camera.",
        )
        parser.add_argument(
            "--rs_frequency",
            type=int,
            default=30,
            help="Frequency for the RealSense camera.",
        )
        parser.add_argument(
            "--visualize_depth",
            action="store_true",
            help="Whether to visualize the depth image.",
        )
        parser.add_argument("--logdir", type=str, help="Directory to load the depth image logdir.")
        return parser

    def __init__(self, args):
        super().__init__("motion_target_publisher")

        self.args = args
        self._parse_config()
        self._start_rs_pipeline()
        self._load_models()

    def _parse_config(self):
        """Parse the configuration of the policy model"""
        with open(os.path.join(self.args.logdir, "params", "env.yaml")) as f:
            self.env_cfg = yaml.unsafe_load(f)
        self.output_resolution = [
            self.env_cfg["scene"]["camera"]["pattern_cfg"]["width"],
            self.env_cfg["scene"]["camera"]["pattern_cfg"]["height"],
        ]
        if (
            self.output_resolution[1] / self.output_resolution[0]
            != self.args.rs_resolution[1] / self.args.rs_resolution[0]
        ):
            self.get_logger().error("Output resolution and RealSense resolution aspect ratios do not match.")
            raise SystemExit()
        self.depth_range = self.env_cfg["scene"]["camera"]["noise_pipeline"]["depth_normalization"]["depth_range"]
        if self.env_cfg["scene"]["camera"]["noise_pipeline"]["depth_normalization"]["normalize"]:
            self.depth_output_range = self.env_cfg["scene"]["camera"]["noise_pipeline"]["depth_normalization"][
                "output_range"
            ]
        else:
            self.depth_output_range = self.depth_range

    def _start_rs_pipeline(self):
        """Start the RealSense camera pipeline."""
        self.rs_pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(
            rs.stream.depth,
            self.args.rs_resolution[0],
            self.args.rs_resolution[1],
            rs.format.z16,
            self.args.rs_frequency,
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

    def _load_models(self):
        """Load the ONNX model for the agent."""
        # load ONNX models
        ort_execution_providers = ort.get_available_providers()
        depth_encoder_path = os.path.join(self.args.logdir, "exported", "0-depth_encoder.onnx")
        self.depth_encoder = ort.InferenceSession(depth_encoder_path, providers=ort_execution_providers)
        print(f"Loaded ONNX models from {self.args.logdir}")

    def start_ros_handlers(self):
        """Start the ROS handlers for the node."""
        self.depth_latent_publisher = self.create_publisher(Float32MultiArray, "depth_latent", 10)
        self.depth_latent_update_period_s = 1 / self.args.publish_frequency
        self.get_logger().info(f"depth_latent_update_period_s: {self.depth_latent_update_period_s} seconds.")
        self.main_timer = self.create_timer(self.depth_latent_update_period_s, self.main_loop_callback)

    def main_loop_callback(self):
        """Main loop callback to publish the depth latent."""
        depth_normalized = self.get_depth_frame()
        depth_normalized = np.expand_dims(np.expand_dims(depth_normalized, 0), 0)
        depth_latent = self.depth_encoder.run(None, {self.depth_encoder.get_inputs()[0].name: depth_normalized})[0][0]
        msg = Float32MultiArray()
        msg.data = depth_latent.tolist()
        self.depth_latent_publisher.publish(msg)
        self.get_logger().info("Start publishing depth latent.", once=True)

    def get_depth_frame(self):
        # read from pyrealsense2, preprocess and write the model latent to the buffer
        rs_frame = self.rs_pipeline.wait_for_frames()  # ms
        depth_frame = rs_frame.get_depth_frame()
        if not depth_frame:
            self.get_logger().error("No depth frame", throttle_duration_sec=1)
            return
        # apply relsense filters
        for rs_filter in self.rs_filters:
            depth_frame = rs_filter.process(depth_frame)
        depth_image_np = np.asanyarray(depth_frame.get_data(), dtype=np.float32) * self.depth_scale

        depth_image_resize = cv2.resize(depth_image_np, (self.output_resolution[0], self.output_resolution[1]))

        filt_m = np.clip(depth_image_resize, self.depth_range[0], self.depth_range[1])
        filt_norm = (filt_m - self.depth_range[0]) / (self.depth_range[1] - self.depth_range[0])

        if self.args.visualize_depth:
            # display normalized depth as 8-bit for visualization
            vis = (filt_norm * 255).astype(np.uint8)
            cv2.imshow("Depth Image", vis)
            # needed to actually render the window and not block forever
            cv2.waitKey(1)

        output_norm = filt_norm * (self.depth_output_range[1] - self.depth_output_range[0]) + self.depth_output_range[0]

        # publish the depth image input to ros topic
        self.get_logger().info("depth range: {}-{}".format(*self.depth_range), once=True)

        return output_norm


def main(args):
    rclpy.init()

    depth_latent_publisher = DepthLatentPublisher(args)
    depth_latent_publisher.start_ros_handlers()

    try:
        rclpy.spin(depth_latent_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Depth Latent Publisher for Unitree robots.")
    parser = DepthLatentPublisher.add_arguments(parser)

    args = parser.parse_args()
    main(args)
