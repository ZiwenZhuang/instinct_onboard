import os
import sys

import numpy as np
import rclpy
import yaml

from instinct_onboard.agents.base import OnboardAgent
from instinct_onboard.agents.shadowing_agent import MotionAsActAgent, ShadowingAgent
from instinct_onboard.robot_cfgs import WirelessButtons
from instinct_onboard.ros_nodes.shadowing import ShadowingNode


class G1ShadowingNode(ShadowingNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.available_agents = dict()
        self.current_agent_name: str | None = None

    def register_agent(self, name: str, agent: OnboardAgent):
        self.available_agents[name] = agent

    def start_ros_handlers(self):
        super().start_ros_handlers()
        main_loop_duration = 0.02
        self.get_logger().info(f"Starting main loop with duration: {main_loop_duration} seconds.")
        self.main_loop_timer = self.create_timer(main_loop_duration, self.main_loop_callback)

    def main_loop_callback(self):

        if self.current_agent_name is None:
            # waiting for motion sequence to arrive
            if self.motion_sequence_buffer is not None:
                self.get_logger().info("Motion sequence received, switching to motion_as_act agent.")
                self.current_agent_name = "motion_as_act"
                self.available_agents[self.current_agent_name].reset()
            else:
                self.get_logger().info("Waiting for motion sequence...", throttle_duration_sec=5.0)
                return
        elif self.current_agent_name == "motion_as_act":
            action, done = self.available_agents[self.current_agent_name].step()
            if done:
                self.get_logger().info(
                    "MotionAsActAgent done, press 'L1' button to switch to shadowing agent.",
                    throttle_duration_sec=10.0,
                )
            if done and (self.joy_stick_buffer.keys & WirelessButtons.L1):
                self.get_logger().info("L1 button pressed, switching to shadowing agent.")
                self.current_agent_name = "shadowing"
                self.available_agents[self.current_agent_name].reset()
            self.send_action(
                action,
                self.available_agents[self.current_agent_name].action_offset,
                self.available_agents[self.current_agent_name].action_scale,
                self.available_agents[self.current_agent_name].p_gains,
                self.available_agents[self.current_agent_name].d_gains,
            )
        elif self.current_agent_name == "shadowing":
            action, done = self.available_agents[self.current_agent_name].step()
            if done:
                self.get_logger().info("ShadowingAgent done.", throttle_duration_sec=5.0)
            self.send_action(
                action,
                self.available_agents[self.current_agent_name].action_offset,
                self.available_agents[self.current_agent_name].action_scale,
                self.available_agents[self.current_agent_name].p_gains,
                self.available_agents[self.current_agent_name].d_gains,
            )


def main(args):
    rclpy.init()

    with open(os.path.join(args.logdir, "params", "env.yaml")) as f:
        cfg = yaml.unsafe_load(f)

    node = G1ShadowingNode(
        cfg=cfg,
        dryrun=not args.nodryrun,
    )

    shadowing_agent = ShadowingAgent(
        logdir=args.logdir,
        ros_node=node,
    )
    motion_as_act_agent = MotionAsActAgent(
        logdir=args.logdir,
        ros_node=node,
        joint_diff_threshold=args.startup_step_size,
        joint_diff_scale=args.startup_step_size,
    )
    node.register_agent("motion_as_act", motion_as_act_agent)
    node.register_agent("shadowing", shadowing_agent)

    node.start_ros_handlers()
    node.get_logger().info("G1ShadowingNode is ready to run.")
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

    parser = argparse.ArgumentParser(description="G1 Shadowing Node")
    parser.add_argument(
        "--logdir",
        type=str,
        help="Directory to load the agent from",
    )
    parser.add_argument(
        "--startup_step_size",
        type=float,
        default=0.2,
        help="Startup step size for the agent (default: 0.2)",
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
