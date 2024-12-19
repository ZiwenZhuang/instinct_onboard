
source ./instinct-venv/bin/activate
source ~/Projects/unitree_ros2/setup.sh
source ~/Codes/ros2_numpy_ws/install/setup.sh
echo "Syncing Time to Jetson"
sshpass -p 123 ssh g1 "(echo 123 | sudo -S timedatectl set-timezone Asia/Shanghai) && (echo 123 | sudo -S date --set=\"$(date '+%Y-%m-%d %H:%M:%S')\")"
