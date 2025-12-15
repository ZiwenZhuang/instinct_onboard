import math

import numpy as np
import rclpy
import ros2_numpy as rnp
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster

from instinct_onboard.ros_nodes.realsense import RsCameraNodeMixin


class RsCamTestNode(RsCameraNodeMixin, Node):
    def __init__(self):
        super().__init__(rs_resolution=(480, 270), rs_fps=60, node_name="rs_cam_test_node")

        self.depth_publisher = self.create_publisher(Image, "/realsense/depth_image", 10)
        self.pointcloud_publisher = self.create_publisher(PointCloud2, "/realsense/pointcloud", 10)
        self.debug_msg_publisher = self.create_publisher(String, "/debug_msg", 10)

        timer_period = 1.0 / self.rs_fps
        self.timer = self.create_timer(timer_period, self.publish_callback)
        self.get_logger().info("RsCamTestNode initialized and ready to publish images.")

        # Publish static TF
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "torso_link"
        t.child_frame_id = "d435_depth_link"
        t.transform.translation.x = 0.04764571478 + 0.0039635 - 0.0042 * math.cos(math.radians(48))
        t.transform.translation.y = 0.015
        t.transform.translation.z = 0.46268178553 - 0.044 + 0.0042 * math.sin(math.radians(48)) + 0.016
        t.transform.rotation.w = math.cos(math.radians(0.5) / 2) * math.cos(math.radians(48) / 2)
        t.transform.rotation.x = math.sin(math.radians(0.5) / 2)
        t.transform.rotation.y = math.sin(math.radians(48) / 2)
        t.transform.rotation.z = 0.0
        self.tf_broadcaster.sendTransform(t)

    def publish_callback(self):
        depth_data = self.refresh_rs_data()
        if depth_data is not None:
            # Publishing depth image may slow down the node, so it is only used in this script.
            depth_msg = self.create_image_msg(depth_data, "mono16")
            self.depth_publisher.publish(depth_msg)
            self.get_logger().info(f"Published depth image of shape {depth_data.shape}", once=True)

            pointcloud_msg = self.create_pointcloud_msg(depth_data)
            self.pointcloud_publisher.publish(pointcloud_msg)
            self.get_logger().info("Published pointcloud", once=True)
        debug_msg = String()
        debug_msg.data = f"Depth data shape: {depth_data.shape}"
        self.debug_msg_publisher.publish(debug_msg)
        self.get_logger().info(debug_msg.data, once=True)

    def create_image_msg(self, data: np.ndarray, encoding: str) -> Image:
        depth_image_msg = rnp.msgify(
            Image, np.asanyarray(data / self.rs_depth_scale, dtype=np.uint16), encoding="16UC1"
        )
        depth_image_msg.header.stamp = self.get_clock().now().to_msg()
        depth_image_msg.header.frame_id = "d435_depth_link"
        return depth_image_msg

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
        msg.header.stamp = self.get_clock().now().to_msg()
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


def main(args=None):
    rclpy.init(args=args)
    node = RsCamTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
