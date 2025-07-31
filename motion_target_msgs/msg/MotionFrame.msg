float32 time_to_target

geometry_msgs/Point pos_b # position target in robot base frame
geometry_msgs/Quaternion quat_w # orientation in world frame
bool[4] pose_mask # plane, height, orientation, heading mask

float32[] dof_pos # joint/dof position
bool[] dof_pos_mask # joint/dof mask

geometry_msgs/Point[] link_pos # link position
bool[] link_pos_mask # link mask
