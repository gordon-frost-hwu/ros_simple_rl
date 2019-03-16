#! /usr/bin/env python
import roslib; roslib.load_manifest('ros_simple_rl')
import rospy
import sys
from auv_msgs.msg import NavSts
from srv_msgs.srv import LogNav, LogNavResponse
global north_list
global east_list
north_list = []
east_list = []

LOG_NAV_SERVICE_TOPIC = "/nav/log_nav"

class NavCSVLogger(object):
    logToFile = False

    def __init__(self):
        sub = rospy.Subscriber("/nav/nav_sts", NavSts, self.navCallback)
        srv = rospy.Service(LOG_NAV_SERVICE_TOPIC, LogNav, self.logNavToFileCallback)

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
            self.logToFile = True   # set flag so that logging in NavSts callback takes place.
            #self.f_north = open("{0}nav_north_data{1}.csv".format(dir_path, unique_identifier_for_files),'w', 1)
            #self.f_east = open("{0}nav_east_data{1}.csv".format(dir_path, unique_identifier_for_files),'w', 1)
            #self.f_yaw = open("{0}nav_yaw_data{1}.csv".format(dir_path, unique_identifier_for_files),'w', 1)

            self.f_comb = open("{0}nav_data{1}.csv".format(dir_path, unique_identifier_for_files),'w', 1)
            self.f_comb.write("#north,east,yaw\n")    #lable columns of log file

        else:
            self.logToFile = False  # set the flag to stop writing to file
            rospy.sleep(0.2)        # wait a while before closing files
            #self.f_north.close()
            #self.f_east.close()
            #self.f_yaw.close()

            self.f_comb.close()
            
        return LogNavResponse()

    def navCallback(self, msg):
        north = msg.position.north
        east = msg.position.east
        yaw = msg.orientation.yaw
        north_str = str(north)+","
        east_str = str(east)+","
        yaw_str = str(yaw)+","
        # Only log when asked to
        if self.logToFile:
            #self.f_north.write(north_str)
            #self.f_east.write(east_str)
            #self.f_yaw.write(yaw_str)

            self.f_comb.write("{0},{1},{2}\n".format(north, east, yaw))

    def on_rospyShutdown(self):
        rospy.loginfo("ROSPY_SHUTDOWN: closing file")
        try:
            #self.f_north.close()
            #self.f_east.close()
            #self.f_yaw.close()
            self.f_comb.close()
        except:
            pass

if __name__ == '__main__':
    print("Node Arguments: ")
    print(sys.argv)
    if "-h" in sys.argv:
            print("USAGE: rosrun q_learning log_nav.py [0|1] [0|1|2|3|N]")
    elif "-c" in sys.argv:
        rospy.init_node('log_of_nav_requester')

        # for boolean argument must convert string of argument into integer first, before bool. Otherwise equals true always
        boolean_flag = bool(int(sys.argv[sys.argv.index("-c") + 1]))
        dir_path = sys.argv[sys.argv.index("-c") + 2]
        descriptor_str = sys.argv[sys.argv.index("-c") + 3]
        call_logging_srv = rospy.ServiceProxy(LOG_NAV_SERVICE_TOPIC, LogNav)
        print("Calling LOG Nav service with given parameters: {0}, {1}".format(boolean_flag, descriptor_str))
        call_logging_srv(log_nav=boolean_flag, dir_path=dir_path, file_name_descriptor=descriptor_str)
    else:
        rospy.init_node('log_of_nav')
        nav_logger = NavCSVLogger()
        rospy.on_shutdown(nav_logger.on_rospyShutdown)
        rospy.spin()
