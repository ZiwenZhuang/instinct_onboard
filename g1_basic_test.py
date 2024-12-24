import os, sys

import rclpy
from rclpy.node import Node

import numpy as np
import yaml

from unitree_hg.msg import LowCmd
from crc_module import get_crc
from unitree_ros2_real import UnitreeRos2Real

class G1Node(UnitreeRos2Real):
    def __init__(self,
            joint_id: int = 0, # the joint id to test
            swing_scale: float = 0.2, # the swing scale in rad
            swing_frequency: float = 0.5, # the swing frequency in Hz
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.joint_id = joint_id
        self.swing_scale = swing_scale
        self.swing_frequency = swing_frequency
        self.anchor_pos = None # The center position of the joint swinging
        self.anchor_time = None # The time when the joint starts swinging

    def parse_config(self):
        return_ = super().parse_config()
        # override observations config
        self.obs_clip = dict()
        self.obs_scales = dict()

        return return_

    def start_ros_handlers(self):
        super().start_ros_handlers()

        print("joint_id:", self.joint_id, "at real idx:", self.dof_map[self.joint_id])
        for i in range(len(self.turn_on_motor_mode)):
            if i == self.joint_id:
                continue
            # to prevent turning off the other motor in control of roll-joint / pitch-joint
            if self.joint_id == 2 and i == 5: continue # waist pitch/roll
            if self.joint_id == 5 and i == 2: continue
            if self.joint_id == 25 and i == 27: continue # left ankle roll/pitch
            if self.joint_id == 27 and i == 25: continue
            if self.joint_id == 26 and i == 28: continue # right ankle roll/pitch
            if self.joint_id == 28 and i == 26: continue
            self.turn_on_motor_mode[i] = 0
            self.p_gains[i] = 0.0001
            self.d_gains[i] = 0.0001

        main_loop_duration = self.cfg["sim"]["dt"] * self.cfg["decimation"]
        print("Starting main loop with duration: ", main_loop_duration)
        self.main_loop_timer = self.create_timer(main_loop_duration, self.main_loop)

        # self.crc_check_subscription = self.create_subscription(
        #     LowCmd,
        #     self.low_cmd_topic,
        #     self._check_lowcmd_crc_callback,
        #     1,
        # )


    def main_loop(self):
        """The single loop for the robot to execute."""
        if self.anchor_pos is None:
            self.anchor_pos = self._get_dof_pos_obs()[self.joint_id]
            self.anchor_time = self.get_clock().now()
        time = self.get_clock().now() - self.anchor_time
        swing = self.swing_scale * np.sin(2 * np.pi * self.swing_frequency * (time.nanoseconds / 1e9))
        self.actions[self.joint_id] = swing
        print("Joint: ", self.joint_id, "Swing: ", swing, end="\r")
        self.send_action(self.actions)

    def _check_lowcmd_crc_callback(self, msg):
        msg_crc = msg.crc
        computed_crc = get_crc(msg)
        print("LowCmd CRC: ", msg_crc, "Computed CRC: ", computed_crc)

def main(args):
    rclpy.init()

    # read params
    with open(os.path.join(args.logdir, "params", "env.yaml"), "r") as f:
        env_cfg = yaml.unsafe_load(f)
    
    # start the node
    node = G1Node(
        joint_id=args.joint_id,
        swing_scale=args.swing_scale,
        swing_frequency=args.swing_frequency,
        cfg=env_cfg,
        dryrun=not args.nodryrun,
    )
    node.start_ros_handlers()

    # start the main loop and let the ROS2 spins
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--joint_id", type= int, default= 0, help= "The joint id to test")
    parser.add_argument("--swing_scale", type= float, default= 0.2, help= "The swing scale in rad")
    parser.add_argument("--swing_frequency", type= float, default= 0.5, help= "The swing frequency in Hz")
    parser.add_argument("--logdir", type= str, default= None, help= "The directory which contains the config.json and model_*.pt files")
    parser.add_argument("--nodryrun", action= "store_true", default= False, help= "Disable dryrun mode")

    args = parser.parse_args()
    main(args)