import sys

import rclpy

import instinct_onboard.robot_cfgs as robot_cfgs
from instinct_onboard.agents.base import ColdStartAgent, OnboardAgent

# from instinct_onboard.agents.parkour_agent import ParkourAgent, ParkourColdStartAgent
from instinct_onboard.agents.parkour_dog_agent import ParkourDogAgent
from instinct_onboard.ros_nodes.parkour import ParkourNode


class Go2ParkourNode(ParkourNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.available_agents = dict()
        self.current_agent_name: str | None = None
        self.parkour_start_flag = False

    def register_agent(self, name: str, agent: OnboardAgent):
        self.available_agents[name] = agent

    def start_ros_handlers(self):
        super().start_ros_handlers()
        main_loop_duration = 0.02
        self.get_logger().info(f"Starting main loop with duration: {main_loop_duration} seconds.")
        self.main_loop_timer = self.create_timer(main_loop_duration, self.main_loop_callback)

    def main_loop_callback(self):
        if self.current_agent_name is None:
            self.get_logger().info("Start cold_start agent.")
            self.current_agent_name = "cold_start"
            self.available_agents[self.current_agent_name].reset()
        elif self.current_agent_name == "cold_start":
            action, done = self.available_agents[self.current_agent_name].step()
            if done:
                self.get_logger().info("ColdStartAgent done.", throttle_duration_sec=5.0)
                if self.joy_stick_buffer.keys & robot_cfgs.WirelessButtons.L1:
                    self.current_agent_name = "parkour"
                    self.available_agents[self.current_agent_name].reset()
            # self.get_logger().info("Start parkour agent.")
            # self.current_agent_name = "parkour"
            self.available_agents[self.current_agent_name].reset()

            self.send_action(
                action,
                self.available_agents[self.current_agent_name].action_offset,
                self.available_agents[self.current_agent_name].action_scale,
                self.available_agents[self.current_agent_name].p_gains,
                self.available_agents[self.current_agent_name].d_gains,
            )
        elif self.current_agent_name == "parkour":
            action, done = self.available_agents[self.current_agent_name].step()
            if done:
                self.get_logger().info("ParkourAgent done.", throttle_duration_sec=5.0)
            if not self.parkour_start_flag:
                self.get_logger().info("Cold start finished. Please press R1 to switch to parkour policy.", once=True)
                action *= 0.0
                if self.joy_stick_buffer.keys & robot_cfgs.WirelessButtons.R1:
                    self.get_logger().info("R1 pressed, stop using cold start agent", once=True)
                    self.parkour_start_flag = True
            self.send_action(
                action,
                self.available_agents[self.current_agent_name].action_offset,
                self.available_agents[self.current_agent_name].action_scale,
                self.available_agents[self.current_agent_name].p_gains,
                self.available_agents[self.current_agent_name].d_gains,
            )


def main(args):
    rclpy.init()

    node = Go2ParkourNode(dryrun=not args.nodryrun, robot_class_name="Go2", imu_state_topic=None)

    parkour_agent = ParkourDogAgent(
        logdir=args.logdir,
        ros_node=node,
    )
    node.register_agent("parkour", parkour_agent)

    # cold_start_agent = ParkourColdStartAgent(
    #     logdir=args.logdir,
    #     dof_max_err=args.dof_max_err,
    #     start_steps=args.start_steps,
    #     ros_node=node,
    # )
    cold_start_agent = ColdStartAgent(
        startup_step_size=1.5,
        ros_node=node,
        joint_target_pos=parkour_agent.action_offset,
        action_scale=parkour_agent.action_scale,
        action_offset=parkour_agent.action_offset,
        p_gains=parkour_agent.p_gains,
        d_gains=parkour_agent.d_gains,
    )
    node.register_agent("cold_start", cold_start_agent)

    node.start_ros_handlers()
    node.get_logger().info("Go2ParkourNode is ready to run.")
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

    parser = argparse.ArgumentParser(description="Go2 Parkour Node")
    parser.add_argument(
        "--logdir",
        type=str,
        help="Directory to load the agent from",
    )
    parser.add_argument(
        "--dof_max_err",
        type=float,
        default=0.25,
        help="Max dof error in start up (default: 0.01)",
    )
    parser.add_argument(
        "--start_steps",
        type=int,
        default=100,
        help="Startup step size for the agent (default: 50)",
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
    if not args.nodryrun:
        args.dof_max_err = 1e10

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
