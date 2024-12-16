# Instinct G1

The code for running network onboard of Unitree G1's Jetson Orin NX

## Prerequisites
- Ubuntu 20.04
- ROS2 Foxy
- unitree_hg ros messages

## Installation
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

4. Some onboard python packages through pip
```bash
pip install pytorch_kinematics numpy-quaternion numpy==1.20.1
```
