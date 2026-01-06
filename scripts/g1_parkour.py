import queue
import sys
import time

import numpy as np
import rclpy
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster

import instinct_onboard.robot_cfgs as robot_cfgs
from instinct_onboard.agents.base import ColdStartAgent
from instinct_onboard.agents.parkour_agent import (
    ParkourAgent,
    ParkourStandAgent,
)
from instinct_onboard.ros_nodes.realsense import UnitreeRsCameraNode

MAIN_LOOP_FREQUENCY_CHECK_INTERVAL = 500


class G1ParkourNode(UnitreeRsCameraNode):
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
        if MAIN_LOOP_FREQUENCY_CHECK_INTERVAL > 1:
            self.main_loop_timer_counter: int = 0  # counter for the main loop timer to assess the actual frequency
            self.main_loop_timer_counter_time = time.time()
            self.main_loop_callback_time_consumptions = queue.Queue(maxsize=MAIN_LOOP_FREQUENCY_CHECK_INTERVAL)

    def main_loop_callback(self):
        main_loop_callback_start_time = time.time()
        if self.current_agent_name is None:
            self.get_logger().info("Starting cold start agent automatically.")
            self.current_agent_name = "cold_start"
            self.available_agents[self.current_agent_name].reset()
            return

        elif self.current_agent_name == "cold_start":
            action, done = self.available_agents[self.current_agent_name].step()
            if done and ("stand" in self.available_agents.keys()):
                self.get_logger().info(
                    "ColdStartAgent done, press 'R1' to switch to stand agent.", throttle_duration_sec=10.0
                )
            else:
                self.get_logger().info(
                    "ColdStartAgent done, press any direction button to switch to parkour agent.",
                    throttle_duration_sec=10.0,
                )
            self.send_action(
                action,
                self.available_agents[self.current_agent_name].action_offset,
                self.available_agents[self.current_agent_name].action_scale,
                self.available_agents[self.current_agent_name].p_gains,
                self.available_agents[self.current_agent_name].d_gains,
            )
            if done and (self.joy_stick_data.R1):
                self.get_logger().info("R1 button pressed, switching to stand agent.")
                self.current_agent_name = "stand"
                self.available_agents[self.current_agent_name].reset()

        elif self.current_agent_name == "stand":
            action, done = self.available_agents[self.current_agent_name].step()
            self.refresh_rs_data()
            self.send_action(
                action,
                self.available_agents[self.current_agent_name].action_offset,
                self.available_agents[self.current_agent_name].action_scale,
                self.available_agents[self.current_agent_name].p_gains,
                self.available_agents[self.current_agent_name].d_gains,
            )
            if self.joy_stick_data.L1:
                self.get_logger().info("L1 button pressed, switching to parkour agent.")
                self.current_agent_name = "parkour"
                self.available_agents[self.current_agent_name].reset()

        elif self.current_agent_name == "parkour":
            action, done = self.available_agents[self.current_agent_name].step()
            self.send_action(
                action,
                self.available_agents[self.current_agent_name].action_offset,
                self.available_agents[self.current_agent_name].action_scale,
                self.available_agents[self.current_agent_name].p_gains,
                self.available_agents[self.current_agent_name].d_gains,
            )
            if self.joy_stick_data.R1:
                self.get_logger().info("R1 button pressed, switching to stand agent.")
                self.current_agent_name = "stand"
                self.available_agents[self.current_agent_name].reset()

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


def main(args):
    rclpy.init()

    node = G1ParkourNode(
        rs_resolution=(480, 270),  # (width, height)
        rs_fps=60,
        camera_individual_process=True,
        joint_pos_protect_ratio=2.0,
        robot_class_name="G1_29Dof",
        dryrun=not args.nodryrun,
    )

    stand_agent = ParkourStandAgent(
        logdir=args.standdir,
        ros_node=node,
    )
    node.register_agent("stand", stand_agent)

    parkour_agent = ParkourAgent(
        logdir=args.logdir,
        ros_node=node,
        depth_vis=args.depth_vis,
        pointcloud_vis=args.pointcloud_vis,
        lin_vel_deadband=args.lin_vel_deadband,
        lin_vel_range=args.lin_vel_range,
        ang_vel_deadband=args.ang_vel_deadband,
        ang_vel_range=args.ang_vel_range,
    )
    node.register_agent("parkour", parkour_agent)

    cold_start_agent = ColdStartAgent(
        startup_step_size=args.startup_step_size,
        ros_node=node,
        joint_target_pos=parkour_agent.default_joint_pos,
        action_scale=parkour_agent.action_scale,
        action_offset=parkour_agent.action_offset,
        p_gains=parkour_agent.p_gains * args.kpkd_factor,
        d_gains=parkour_agent.d_gains * args.kpkd_factor,
    )
    node.register_agent("cold_start", cold_start_agent)

    node.start_ros_handlers()
    node.get_logger().info("G1ParkourNode is ready to run.")
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

    parser = argparse.ArgumentParser(description="G1 Parkour Node")
    parser.add_argument(
        "--standdir",
        type=str,
        help="Directory to load the stand agent from",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        help="Directory to load the parkour agent from",
    )
    parser.add_argument(
        "--startup_step_size",
        type=float,
        default=0.2,
        help="Startup step size for the cold start agent (default: 0.2)",
    )
    parser.add_argument(
        "--kpkd_factor",
        type=float,
        default=2.0,
        help="KPKD factor for the cold start agent (default: 2.0)",
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
        "--lin_vel_deadband",
        type=float,
        default=0.5,
        help="Deadband of wireless control for linear velocity (default: 0.5)",
    )
    parser.add_argument(
        "--lin_vel_range",
        type=list,
        default=[0.5, 0.5],
        help="Range of linear velocity, only forward (default: [0.5 0.5])",
    )
    parser.add_argument(
        "--ang_vel_deadband",
        type=float,
        default=0.5,
        help="Deadband of wireless control for angular velocity (default: 0.5)",
    )
    parser.add_argument(
        "--ang_vel_range",
        type=list,
        default=[0.0, 1.0],
        help="Range of linear velocity, both turn left and turn right (default: [0.0 1.0])",
    )
    parser.add_argument(
        "--nodryrun",
        action="store_true",
        default=False,
        help="Run the node without dry run mode (default: False)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode (default: False)",
    )

    args = parser.parse_args()
    if args.debug:
        # import typing; typing.TYPE_CHECKING = True
        import debugpy

        ip_address = ("0.0.0.0", 6789)
        print("Process: " + " ".join(sys.argv[:]))
        print("Is waiting for attach at address: %s:%d" % ip_address, flush=True)
        debugpy.listen(ip_address)
        debugpy.wait_for_client()
        debugpy.breakpoint()

    main(args)
