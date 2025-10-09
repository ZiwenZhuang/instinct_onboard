from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient

ChannelFactoryInitialize(0)
rsc = RobotStateClient()
rsc.SetTimeout(3.0)
rsc.Init()
code = rsc.ServiceSwitch("sport_mode", False)
if code != 0:
    raise Exception("ServiceSwitch failed with code: %d" % code)
else:
    print("Stop sports mode succeeded")
