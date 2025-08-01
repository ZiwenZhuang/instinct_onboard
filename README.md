# Instinct G1

## Jetson
The code for running network onboard of Unitree G1's Jetson Orin NX

### Prerequisites
- Ubuntu 20.04
- ROS2 Foxy
- unitree_hg ros messages
- unitree_go ros messages (for wireless controller)

### Installation
1. JetPack
```bash
sudo apt-get update
sudo apt install nvidia-jetpack
```

2. python virtual environment
```bash
sudo apt-get install python3-venv
python3 -m venv instinct_venv
source instinct_venv/bin/activate
```

3. Install crc module
Follow the instruction of [crc_module](https://github.com/ZiwenZhuang/g1_crc)

4. Make sure mcap storage for ros2 installed
```bash
sudo apt install ros-{ROS_VERSION}-rosbag2-storage-mcap
```

5. Some onboard python packages through pip
```bash
pip install numpy-quaternion numpy==1.24.4 transformations==2022.9.26
```
