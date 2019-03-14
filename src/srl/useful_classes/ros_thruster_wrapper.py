#! /usr/bin/python
import roslib; roslib.load_manifest("rl_pybrain")
import rospy

from vehicle_interface.srv import DictionaryService
from diagnostic_msgs.msg import KeyValue

class Thrusters(object):
    def __init__(self):
        self.srv_fault = rospy.ServiceProxy('/thrusters/faults', DictionaryService)

    def inject_surge_thruster_fault(self, new_surge_health, index_of_fault):
        """
        :param new_surge_health: 17 is 20% health, 85 is 100% health
        :param index_of_fault: (fwd-port, fwd-std, lat-rear, lat-front, vert-rear, vert-front)
        :return:
        """
        fault_type = KeyValue(key='fault_type',value='hard_limit')
        print("Injecting Fault of {0} into thruster {1}".format(new_surge_health, index_of_fault))
        if index_of_fault == 0:
            th_min = KeyValue(key='th_min', value='-{0}, -85, -85, -85, -85, -85'.format(str(new_surge_health)))
            th_max = KeyValue(key='th_max', value='{0}, 85, 85, 85, 85, 85'.format(str(new_surge_health)))
        elif index_of_fault == 1:
            th_min = KeyValue(key='th_min', value='-85, -{0}, -85, -85, -85, -85'.format(str(new_surge_health)))
            th_max = KeyValue(key='th_max', value='85, {0}, 85, 85, 85, 85'.format(str(new_surge_health)))

        thruster_fault_response = self.srv_fault(request=[fault_type, th_min, th_max])

        last_thruster_failure_index = index_of_fault
        last_thruster_health = new_surge_health
        print("Thruster Health changed to:\n{0}".format(thruster_fault_response))
        print(thruster_fault_response.response[-1].value)