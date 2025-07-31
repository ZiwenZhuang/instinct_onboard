# Instinct G1

## Jetson
The code for running network onboard of Unitree G1's Jetson Orin NX

### Prerequisites
- Ubuntu 20.04
- ROS2 Foxy
- unitree_hg ros messages
- unitree_go ros messages (for wireless controller)

### Installation
- JetPack
    ```bash
    sudo apt-get update
    sudo apt install nvidia-jetpack
    ```

- Install crc module

    Follow the instruction of [crc_module](https://github.com/ZiwenZhuang/g1_crc) and copy the product (`crc_module.so`) to where you launch the python script.

- Make sure mcap storage for ros2 installed
    ```bash
    sudo apt install ros-{ROS_VERSION}-rosbag2-storage-mcap
    ```

- Install ROS2 packages in your own ROS workspace.

- python virtual environment
    ```bash
    sudo apt-get install python3-venv
    python3 -m venv instinct_venv
    source instinct_venv/bin/activate
    ```

- Some onboard python packages through pip
    ```bash
    pip install -e .
    ```

## Code Structure Introduction

### ROS nodes

- In `instinct_onboard/ros_nodes/`, you can find the ROS nodes that are used to communicate with the robot.

- To avoid diamond inheritance, each function-specific ROS node should be implemented in a dedicated file with Mixin class.

- Please inherit everything you need in the script as well as the state machine logic in your main-entry script. (in `scripts/`)

### Agents

- In `instinct_onboard/agents/`, you can find the agents that are used to run the network (as well as collect the observations).

- Do NOT scale the action of the network output. The action scaling happens in the ros node side.
