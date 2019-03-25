#! /usr/bin/env python
import roslib; roslib.load_manifest('ros_simple_rl')
import rospy
import sys
from auv_msgs.msg import NavSts
from std_msgs.msg import Bool
from srv_msgs.srv import LogNav, LogNavResponse
from vehicle_interface.msg import PilotRequest

# sync signal start
# pilot request for X seconds
# sync signal end
LOG_NAV_SERVICE_TOPIC = "/nav/log_nav"

class ControlResponse(object):
    def __init__(self):
        pilot = rospy.Publisher("/pilot/velocity_req", PilotRequest, queue_size=1)
        
if __name__ == '__main__':
    rospy.init_node("control_response")
    
    pilot = rospy.Publisher("/pilot/velocity_req", PilotRequest, queue_size=1)
    signal = rospy.Publisher("/response_signal", Bool, queue_size=1)
    
    call_logging_srv = rospy.ServiceProxy(LOG_NAV_SERVICE_TOPIC, LogNav)
    print("Calling LOG Nav service with given parameters: {0}, {1}".format(True, "HA"))
    call_logging_srv(log_nav=True, dir_path="/tmp/", file_name_descriptor="ControlResponse")
    
    pilotRequest = PilotRequest()
    pilotRequest.velocity = [0, 0, 0, 0, 0, 1]
    
    rate = rospy.Rate(10)
    signalTrue = Bool()
    signalTrue.data = True
    
    
    count = 0
    while not rospy.is_shutdown():
        signal.publish(signalTrue)
        pilot.publish(pilotRequest)
        if count > 100:
            break
        count = count + 1
        rate.sleep()
    signalFalse = Bool()
    signalFalse.data = False
    signal.publish(signalFalse)
    print("Stopping nav recording")
    call_logging_srv(log_nav=False, dir_path="/tmp/", file_name_descriptor="ControlResponse")
    rospy.spin()
