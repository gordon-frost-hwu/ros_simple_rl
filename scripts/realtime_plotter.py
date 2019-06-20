#!/usr/bin/env python
import roslib; roslib.load_manifest("ros_simple_rl")

import numpy as np
import matplotlib.pyplot as plt
import rospy
from auv_msgs.msg import NavSts
import sys

class RealTimePlotter(object):
    def __init__(self, args):
        self.args = args
        self.active_args = {}
        self.parse_args()
        print(self.active_args)
        self.values = []

        self.fig, self.axes = plt.subplots()
        rospy.on_shutdown(self.on_rospy_shutdown)

        rospy.Subscriber("/nav/nav_sts", NavSts, self.navCallback)

        # self.axes = plt.gca()
        # axes.set_xlim([0.0, xmax])
        ymin = -np.pi
        ymax = np.pi
        if "ymin" in self.active_args:
            ymin = float(self.active_args["ymin"])
        if "ymax" in self.active_args:
            ymax = float(self.active_args["ymax"])
            print(ymin)
            print(ymax)
        self.axes.set_ylim([ymin, ymax])
        self.win = self.fig.canvas.manager.window
        self.win.after(500, self.animate)
        plt.show()

    def animate(self):
        #for (tim, val) in self.buffer:
         #   self.axes.plot(tim, val)
        plt.draw()
        print("Animating")
        self.win.after(500, self.animate)

    def navCallback(self, msg):
        stamp = msg.header.stamp
        time = stamp.secs + stamp.nsecs * 1e-9

        value = msg.orientation.yaw
        self.values.append((time, value))

        self.axes.plot(time, value, ".")

    def getArg(self, key):
        if key in self.args:
            key_idx = self.args.index(key)
            value = self.args[key_idx + 1]
            return key.strip("-"), value
        return None, None

    def parse_args(self):
        if len(self.args) == 0:
            return
        for arg in self.args:
            if arg[0] == "-" and arg[1].isalpha():
                key, val = self.getArg(arg)
                if key is not None:
                    self.active_args[key] = val
        print("Enabled arguments: {0}".format(self.active_args))

    def on_rospy_shutdown(self):
        if "f" in self.active_args:
            filename = self.active_args["f"]
            print("Saving values to file: {0}".format(filename))

            with open(filename, 'wb') as csv_file:
                for (time, val) in self.values:
                    csv_file.write("{0}\t{1}".format(time, val))
                    csv_file.write(('\n'))


if __name__ == '__main__':
    rospy.init_node("plotter")

    args = sys.argv
    print("Args: {0}".format(args))
    #args.pop(0)
    #plt.ion()
    plotter = RealTimePlotter(args)

    rospy.spin()