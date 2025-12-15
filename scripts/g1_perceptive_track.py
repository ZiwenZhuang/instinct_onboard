import os
import queue
import sys
import time

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster

from instinct_onboard.agents.base import ColdStartAgent
from instinct_onboard.agents.tracking_agent import PerceptiveTrackerAgent, TrackerAgent
from instinct_onboard.agents.walk_agent import WalkAgent
from instinct_onboard.robot_cfgs import WirelessButtons
from instinct_onboard.ros_nodes.realsense import UnitreeRsCameraNode

MAIN_LOOP_FREQUENCY_CHECK_INTERVAL = 500


class G1TrackingNode(UnitreeRsCameraNode):
    def __init__(self, *args, motion_vis: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.available_agents = dict()
        self.current_agent_name: str | None = None
        self.motion_vis = motion_vis

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
        if MAIN_LOOP_FREQUENCY_CHECK_INTERVAL > 1:
            self.main_loop_timer_counter: int = 0  # counter for the main loop timer to assess the actual frequency
            self.main_loop_timer_counter_time = time.time()
            self.main_loop_callback_time_consumptions = queue.Queue(maxsize=MAIN_LOOP_FREQUENCY_CHECK_INTERVAL)
        # start the visualization timer with 100ms duration
        vis_duration = 0.1
        if self.motion_vis:
            self.vis_timer = self.create_timer(vis_duration, self.vis_callback)

    def main_loop_callback(self):
        main_loop_callback_start_time = time.time()
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
            if done and ("walk" in self.available_agents.keys()):
                self.get_logger().info(
                    "ColdStartAgent done, press 'L1' to switch to walk agent.", throttle_duration_sec=10.0
                )
            else:
                self.get_logger().info(
                    "ColdStartAgent done, press any direction button to switch to tracking agent.",
                    throttle_duration_sec=10.0,
                )
            self.send_action(
                action,
                self.available_agents[self.current_agent_name].action_offset,
                self.available_agents[self.current_agent_name].action_scale,
                self.available_agents[self.current_agent_name].p_gains,
                self.available_agents[self.current_agent_name].d_gains,
            )
            if done and (self.joy_stick_buffer.keys & WirelessButtons.L1):
                self.get_logger().info("L1 button pressed, switching to walk agent.")
                self.current_agent_name = "walk"
                self.available_agents[self.current_agent_name].reset()
            if done and (self.joy_stick_buffer.keys & WirelessButtons.up):
                if "walk" in self.available_agents.keys():
                    self.get_logger().warn("up button pressed, but there is a walk agent registered. ignored")

        elif self.current_agent_name == "walk":
            action, done = self.available_agents[self.current_agent_name].step()
            self.send_action(
                action,
                self.available_agents[self.current_agent_name].action_offset,
                self.available_agents[self.current_agent_name].action_scale,
                self.available_agents[self.current_agent_name].p_gains,
                self.available_agents[self.current_agent_name].d_gains,
            )
            if self.joy_stick_buffer.keys & WirelessButtons.up:
                self.get_logger().info("up button pressed, switching to tracking agent.")
                self.current_agent_name = "tracking"
                self.available_agents[self.current_agent_name].reset("diveroll4-ziwen-0-retargeted.npz")
            elif self.joy_stick_buffer.keys & WirelessButtons.down:
                self.get_logger().info("down button pressed, switching to tracking agent.")
                self.current_agent_name = "tracking"
                self.available_agents[self.current_agent_name].reset("kneelClimbStep1-x-0.1-ziwen-retargeted.npz")
            elif self.joy_stick_buffer.keys & WirelessButtons.left:
                self.get_logger().info("left button pressed, switching to tracking agent.")
                self.current_agent_name = "tracking"
                self.available_agents[self.current_agent_name].reset("rollVault11-ziwen-retargeted.npz")
            elif self.joy_stick_buffer.keys & WirelessButtons.right:
                self.get_logger().info("right button pressed, switching to tracking agent.")
                self.current_agent_name = "tracking"
                self.available_agents[self.current_agent_name].reset("jumpsit2-ziwen-retargeted.npz")
            elif self.joy_stick_buffer.keys & WirelessButtons.X:
                self.get_logger().info("right button pressed, switching to tracking agent.")
                self.current_agent_name = "tracking"
                self.available_agents[self.current_agent_name].reset("superheroLanding-retargeted.npz")

        elif self.current_agent_name == "tracking":
            action, done = self.available_agents[self.current_agent_name].step()
            self.send_action(
                action,
                self.available_agents[self.current_agent_name].action_offset,
                self.available_agents[self.current_agent_name].action_scale,
                self.available_agents[self.current_agent_name].p_gains,
                self.available_agents[self.current_agent_name].d_gains,
            )
            if self.joy_stick_buffer.keys & WirelessButtons.L1:
                self.get_logger().info(
                    "L1 button pressed, switching to walk agent (no matter whether the tracking agent is done)."
                )
                self.current_agent_name = "walk"
                self.available_agents[self.current_agent_name].reset()
            if done and ("walk" in self.available_agents.keys()):
                # switch to walk agent
                self.get_logger().info("TrackingAgent done, switching to walk agent.")
                self.current_agent_name = "walk"
                self.available_agents[self.current_agent_name].reset()
            elif done:
                self.get_logger().info("TrackingAgent done, turning off motors.")
                self._turn_off_motors()
                sys.exit(0)

        # count the main loop timer counter and log the actual frequency every 500 counts
        if MAIN_LOOP_FREQUENCY_CHECK_INTERVAL > 1:
            self.main_loop_callback_time_consumptions.put(time.time() - main_loop_callback_start_time)
            self.main_loop_timer_counter += 1
            if self.main_loop_timer_counter % MAIN_LOOP_FREQUENCY_CHECK_INTERVAL == 0:
                time_consumptions = [
                    self.main_loop_callback_time_consumptions.get() for _ in range(MAIN_LOOP_FREQUENCY_CHECK_INTERVAL)
                ]
                self.get_logger().info(
                    f"Actual main loop frequency: {(MAIN_LOOP_FREQUENCY_CHECK_INTERVAL / (time.time() - self.main_loop_timer_counter_time)):.2f} Hz. Mean time consumption: {np.mean(time_consumptions):.4f} s."
                )
                self.main_loop_timer_counter = 0
                self.main_loop_timer_counter_time = time.time()

    def vis_callback(self):
        agent: PerceptiveTrackerAgent = self.available_agents["tracking"]
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
        rs_resolution=(480, 270),  # (width, height)
        rs_fps=60,
        camera_individual_process=True,
        joint_pos_protect_ratio=2.0,
        robot_class_name="G1_29Dof",
        motion_vis=args.motion_vis,
        dryrun=not args.nodryrun,
    )

    tracking_agent = PerceptiveTrackerAgent(
        logdir=args.logdir,
        motion_file_dir=args.motion_dir,
        depth_vis=args.depth_vis,
        pointcloud_vis=args.pointcloud_vis,
        ros_node=node,
    )
    if args.walk_logdir is not None:
        walk_agent = WalkAgent(
            logdir=args.walk_logdir,
            ros_node=node,
        )
        cold_start_agent = ColdStartAgent(
            startup_step_size=args.startup_step_size,
            ros_node=node,
            joint_target_pos=walk_agent.default_joint_pos,
            action_scale=walk_agent.action_scale,
            action_offset=walk_agent.action_offset,
            p_gains=walk_agent.p_gains * args.kpkd_factor,
            d_gains=walk_agent.d_gains * args.kpkd_factor,
        )
        node.register_agent("walk", walk_agent)
    else:
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
        help="Directory to the motion files",
    )
    parser.add_argument(
        "--walk_logdir",
        type=str,
        help="Directory to load the walk agent from",
        default=None,
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
        default=2.0,
        help="KPKD factor for the cold start agent (default: 2.0)",
    )
    parser.add_argument(
        "--motion_vis",
        action="store_true",
        default=False,
        help="Visualize the motion sequence by publishing motion sequence as joint state, need robot state publisher to visuzlize the robot model (default: False)",
    )
    parser.add_argument(
        "--depth_vis",
        action="store_true",
        default=False,
        help="Visualize the depth image (default: False)",
    )
    parser.add_argument(
        "--pointcloud_vis",
        action="store_true",
        default=False,
        help="Visualize the pointcloud (default: False)",
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
