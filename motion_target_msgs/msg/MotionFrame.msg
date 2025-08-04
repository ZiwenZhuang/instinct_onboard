float32 time_to_target

geometry_msgs/Point pos_b # position target in robot base frame
geometry_msgs/Quaternion quat_w # orientation in world frame, all zero shows invalid orientation
bool[4] pose_mask # plane, height, orientation, heading mask

float32[] joint_pos # joint/dof position
bool[] joint_pos_mask # joint/dof mask

geometry_msgs/Point[] link_pos # link position in base frame
bool[] link_pos_mask # link mask

geometry_msgs/Quaternion[] link_quat # link orientation in base frame
bool[] link_quat_mask # link orientation mask
