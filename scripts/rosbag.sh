#!/bin/bash
ros2 bag record -s mcap -o $1 \
    /rosout /tf /tf_static /tf2_web_republisher/tf /tf2_web_republisher/tf_static /parameter_events \
    /lowstate /lowcmd /secondary_imu /lowstate_doubleimu /wirelesscontroller /lf/bmsstate \
    /motion_target /raw_actions
# Usage: ./rosbag.sh <bag_name>
# Example: ./rosbag.sh my_bag
