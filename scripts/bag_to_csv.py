#! /usr/bin/env python
import roslib;

roslib.load_manifest('ros_simple_rl')
import rospy
import sys
import math

# ROS Msgs
from auv_msgs.msg import NavSts
from srv_msgs.srv import LogNav, LogNavResponse
from my_msgs.msg import BehavioursGoal
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage
from vehicle_interface.msg import PilotRequest
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion

LOG_NAV_SERVICE_TOPIC = "/nav/log_nav"

INDEX_TIMESTAMP = 0
INDEX_NAV_YAW = 1
INDEX_NAV_YAWRATE = 2
INDEX_TF_ROLL = 3
INDEX_TF_PITCH = 4
INDEX_TF_YAW = 5
INDEX_COMPASS_ROLL = 6
INDEX_COMPASS_PITCH = 7
INDEX_COMPASS_YAW = 8
INDEX_PILOTREQ_YAW = 9
INDEX_BEHAVIOURS_GOAL_YAW = 10

TOPIC_NAV = "/nav/nav_sts"
TOPIC_TF = "/tf"
TOPIC_COMPASS = "/compass/orientation"
TOPIC_PILOT_REQUEST = "/pilot/position_req"
TOPIC_BEHAVIOURS_GOAL = "/behaviours/goal"


class CSVLogger(object):
    logToFile = True

    def __init__(self, filename):
        sub1 = rospy.Subscriber(TOPIC_NAV, NavSts, self.navCallback)
        sub2 = rospy.Subscriber(TOPIC_TF, TFMessage, self.tfCallback)
        sub3 = rospy.Subscriber(TOPIC_COMPASS, Imu, self.compassCallback)
        sub4 = rospy.Subscriber(TOPIC_PILOT_REQUEST, PilotRequest, self.pilotRequestCallback)
        sub5 = rospy.Subscriber(TOPIC_BEHAVIOURS_GOAL, BehavioursGoal, self.behavioursGoalCallback)

        # srv = rospy.Service(LOG_NAV_SERVICE_TOPIC, LogNav, self.logNavToFileCallback)

        directory = "/tmp/"
        filepath = "{0}{1}".format(directory, filename)
        print("Creating file: {0}".format(filepath))
        self.f_comb = open(filepath, 'w', 1)
        self.f_comb.write("#timestamp\tnav_yaw\tnav_yaw_rate\ttf_roll\ttf_pitch\ttf_yaw\tcompass_roll\tcompass_pitch\tcompass_yaw"
                          "\tpilot_request_yaw\tbehaviours_goal_yaw\n")  # label columns of log file

        self.record = {INDEX_TIMESTAMP: 0, INDEX_BEHAVIOURS_GOAL_YAW: 0, INDEX_COMPASS_YAW: 0,
                       INDEX_NAV_YAW: 0, INDEX_PILOTREQ_YAW: 0, INDEX_TF_YAW: 0, INDEX_NAV_YAWRATE: 0,
                       INDEX_TF_ROLL: 0, INDEX_TF_PITCH: 0, INDEX_COMPASS_ROLL: 0, INDEX_COMPASS_PITCH: 0}

    def tfCallback(self, msg):
        quaternion = msg.transforms[0].transform.rotation
        roll, pitch, yaw = self.quaternion_to_euler(quaternion.x, quaternion.y, quaternion.z, quaternion.w)

        self.record[INDEX_TF_ROLL] = roll
        self.record[INDEX_TF_PITCH] = pitch
        self.record[INDEX_TF_YAW] = yaw

        stamp = msg.transforms[0].header.stamp
        self.record[INDEX_TIMESTAMP] = stamp.secs + (stamp.nsecs / 1e9)

        self.f_comb.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\n".format(self.record[0],
                                                 self.record[1],
                                                 self.record[2],
                                                 self.record[3],
                                                 self.record[4],
                                                 self.record[5],
                                                 self.record[6],
                                                 self.record[7],
                                                 self.record[8],
                                                 self.record[9],
                                                 self.record[10]
                                                 ))

    def compassCallback(self, msg):
        quaternion = msg.orientation
        roll, pitch, yaw = self.quaternion_to_euler(quaternion.x, quaternion.y, quaternion.z, quaternion.w)
        self.record[INDEX_COMPASS_ROLL] = roll
        self.record[INDEX_COMPASS_PITCH] = pitch
        self.record[INDEX_COMPASS_YAW] = yaw

    def pilotRequestCallback(self, msg):
        self.record[INDEX_PILOTREQ_YAW] = msg.position[5]

    def behavioursGoalCallback(self, msg):
        self.record[INDEX_BEHAVIOURS_GOAL_YAW] = msg.goal_velocity.values[5]

    def navCallback(self, msg):
        yaw = msg.orientation.yaw
        self.record[INDEX_NAV_YAW] = yaw

        yaw_rate = msg.orientation_rate.yaw
        self.record[INDEX_NAV_YAWRATE] = yaw_rate

    def quaternion_to_euler(self, x, y, z, w):

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
        return [yaw, pitch, roll]

    def on_rospyShutdown(self):
        rospy.loginfo("ROSPY_SHUTDOWN: closing file")
        try:
            # self.f_north.close()
            # self.f_east.close()
            # self.f_yaw.close()
            self.f_comb.close()
        except:
            pass

    def logNavToFileCallback(self, req):
        # Open unique files to log the data to
        unique_identifier_for_files = req.file_name_descriptor
        dir_path = req.dir_path
        if dir_path[-1] == "/":
            pass
        else:
            dir_path = "{0}/".format(dir_path)
        # open the logging file; 1 is for line buffering.
        if req.log_nav:
            self.logToFile = True  # set flag so that logging in NavSts callback takes place.
            # self.f_north = open("{0}nav_north_data{1}.csv".format(dir_path, unique_identifier_for_files),'w', 1)
            # self.f_east = open("{0}nav_east_data{1}.csv".format(dir_path, unique_identifier_for_files),'w', 1)
            # self.f_yaw = open("{0}nav_yaw_data{1}.csv".format(dir_path, unique_identifier_for_files),'w', 1)



        else:
            self.logToFile = False  # set the flag to stop writing to file
            rospy.sleep(0.2)  # wait a while before closing files
            # self.f_north.close()
            # self.f_east.close()
            # self.f_yaw.close()

            self.f_comb.close()

        return LogNavResponse()


if __name__ == '__main__':
    rospy.init_node('bag_to_csv')
    args = sys.argv
    if len(args) != 2:
        print("usage: python bag_to_csv.py [filename]")
        exit(0)
    filename = args[1]

    csv_logger = CSVLogger(filename)
    rospy.on_shutdown(csv_logger.on_rospyShutdown)
    rospy.spin()
