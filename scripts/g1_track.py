import os
import sys

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster

from instinct_onboard.agents.base import ColdStartAgent
from instinct_onboard.agents.tracking_agent import TrackerAgent
from instinct_onboard.robot_cfgs import WirelessButtons
from instinct_onboard.ros_nodes.unitree import UnitreeNode


class G1TrackingNode(UnitreeNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.available_agents = dict()
        self.current_agent_name: str | None = None

    def register_agent(self, name: str, agent):
        self.available_agents[name] = agent

    def start_ros_handlers(self):
        super().start_ros_handlers()
        # build the joint state publisher and base_link tf publisher
        self.joint_state_publisher = self.create_publisher(JointState, "joint_states", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        # start the main loop with 20ms duration
        main_loop_duration = 0.02
        self.get_logger().info(f"Starting main loop with duration: {main_loop_duration} seconds.")
        self.main_loop_timer = self.create_timer(main_loop_duration, self.main_loop_callback)
        # start the visualization timer with 100ms duration
        vis_duration = 0.1
        self.vis_timer = self.create_timer(vis_duration, self.vis_callback)

    def main_loop_callback(self):
        if self.current_agent_name is None:
            self.get_logger().info("Starting cold start agent automatically.")
            self.get_logger().info("Press 'A' button to match motion to current heading.", throttle_duration_sec=2.0)
            self.current_agent_name = "cold_start"
            self.available_agents[self.current_agent_name].reset()
            return

        if self.joy_stick_buffer.keys & WirelessButtons.A:
            self.get_logger().info("A button pressed, matching motion to current heading.", throttle_duration_sec=2.0)
            self.available_agents["tracking"].match_to_current_heading()

        elif self.current_agent_name == "cold_start":
            action, done = self.available_agents[self.current_agent_name].step()
            if done:
                self.get_logger().info(
                    "ColdStartAgent done, press 'L1' to switch to tracking agent.", throttle_duration_sec=10.0
                )
            self.send_action(
                action,
                self.available_agents[self.current_agent_name].action_offset,
                self.available_agents[self.current_agent_name].action_scale,
                self.available_agents[self.current_agent_name].p_gains,
                self.available_agents[self.current_agent_name].d_gains,
            )
            if done and (self.joy_stick_buffer.keys & WirelessButtons.L1):
                self.get_logger().info("L1 button pressed, switching to tracking agent.")
                self.current_agent_name = "tracking"
                self.available_agents[self.current_agent_name].reset()

        elif self.current_agent_name == "tracking":
            action, done = self.available_agents[self.current_agent_name].step()
            self.send_action(
                action,
                self.available_agents[self.current_agent_name].action_offset,
                self.available_agents[self.current_agent_name].action_scale,
                self.available_agents[self.current_agent_name].p_gains,
                self.available_agents[self.current_agent_name].d_gains,
            )
            if done:
                self.get_logger().info("TrackingAgent done, turning off motors.")
                self._turn_off_motors()
                sys.exit(0)

    def vis_callback(self):
        agent: TrackerAgent = self.available_agents["tracking"]
        cursor = agent.motion_cursor_idx
        # Publish JointState for target joints
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = self.sim_joint_names
        joint_pos = agent.motion_data.joint_pos[cursor]
        joint_vel = agent.motion_data.joint_vel[cursor]
        js.position = joint_pos.tolist()
        js.velocity = joint_vel.tolist()
        js.effort = [0.0] * len(joint_pos)
        self.joint_state_publisher.publish(js)
        # Broadcast TF for target base
        pos = agent.motion_data.base_pos[cursor]
        quat = agent.motion_data.base_quat[cursor]
        t = TransformStamped()
        t.header.stamp = js.header.stamp
        t.header.frame_id = "world"
        t.child_frame_id = "torso_link"
        t.transform.translation.x = float(pos[0])
        t.transform.translation.y = float(pos[1])
        t.transform.translation.z = float(pos[2])
        t.transform.rotation.w = float(quat[0])
        t.transform.rotation.x = float(quat[1])
        t.transform.rotation.y = float(quat[2])
        t.transform.rotation.z = float(quat[3])
        self.tf_broadcaster.sendTransform(t)


def main(args):
    rclpy.init()

    node = G1TrackingNode(
        robot_class_name="G1_29Dof",
        dryrun=not args.nodryrun,
    )

    tracking_agent = TrackerAgent(
        logdir=args.logdir,
        motion_file_dir=args.motion_dir,
        ros_node=node,
    )
    cold_start_agent = tracking_agent.get_cold_start_agent(args.startup_step_size, args.kpkd_factor)

    node.register_agent("cold_start", cold_start_agent)
    node.register_agent("tracking", tracking_agent)

    node.start_ros_handlers()
    node.get_logger().info("G1TrackingNode is ready to run.")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("Node shutdown complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="G1 Tracking Node")
    parser.add_argument(
        "--logdir",
        type=str,
        help="Directory to load the agent from",
    )
    parser.add_argument(
        "--motion_dir",
        type=str,
        help="Path to the motion file",
    )
    parser.add_argument(
        "--startup_step_size",
        type=float,
        default=0.2,
        help="Startup step size for the cold start agent (default: 0.2)",
    )
    parser.add_argument(
        "--nodryrun",
        action="store_true",
        default=False,
        help="Run the node without dry run mode (default: False)",
    )
    parser.add_argument(
        "--kpkd_factor",
        type=float,
        default=1.0,
        help="KPKD factor for the cold start agent (default: 1.0)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode (default: False)",
    )

    args = parser.parse_args()

    if args.debug:
        import debugpy

        ip_address = ("0.0.0.0", 6789)
        print("Process: " + " ".join(sys.argv[:]))
        print("Is waiting for attach at address: %s:%d" % ip_address, flush=True)
        debugpy.listen(ip_address)
        debugpy.wait_for_client()
        debugpy.breakpoint()

    main(args)
