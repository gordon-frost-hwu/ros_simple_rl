#! /usr/bin/python
import roslib
roslib.load_manifest("ros_simple_rl")

import time
from datetime import datetime
from enum import Enum, IntEnum
import os
import rospy
import sys
import argparse
import numpy as np
from ga_optimizer import GAOptimizer
from processes.pilot_pid_process import PilotPidProcess
from srl.environments.ros_behaviour_interface import ROSBehaviourInterface

PROP_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]

class Level(IntEnum):
    INFO = 1
    DEBUG = 2
    TRACE = 3

class SimpleLogger(object):
    def __init__(self, level=Level.INFO):
        self._level = level
        self._flush_every = 5
        self._entries_since_last_flush = 0
        self._log_file = None

    def initialise(self, dir_path):
        dir_path = os.path.join(dir_path, '')
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = "log-{0}.log".format(timestamp)
        self._log_file = open("{0}{1}".format(dir_path, filename), "w+")

    def debug(self, s):
        if Level.DEBUG <= self._level:
            self._log("DEBUG", s)

    def info(self, s):
        if Level.INFO <= self._level:
            self._log("INFO", s)

    def trace(self, s):
        if Level.TRACE <= self._level:
            self._log("TRACE", s)

    def _log(self, type, s):
        timestamp = datetime.now().time()
        timestamp_str = timestamp.strftime("%H:%M:%S.%f")

        if s[-1] is "\n":
            log_entry = "{0}\t{1}\t{2}".format(timestamp_str, type, s)
            if self._log_file is not None:
                self._log_file.write(log_entry)
        else:
            s = "{0}{1}".format(s, "\n")
            log_entry = "{0}\t{1}\t{2}".format(timestamp_str, type, s)
            if self._log_file is not None:
                self._log_file.write(log_entry)
        if self._log_file is not None:
            self._entries_since_last_flush += 1

            if self._entries_since_last_flush == self._flush_every - 1:
                self._log_file.flush()

        # Always print to console
        print(log_entry[0:-1])


class DirectoryManager(object):
    def __init__(self, logger, root_directory):
        self.name = "DirectoryManager"
        self._logger = logger

        if not os.path.exists(root_directory):
            os.mkdir(root_directory)
            self.log("{0} - created the root directory: {1}".format(self.name, root_directory))
        else:
            self.log("{0} - using the existing root directory: {1}".format(self.name, root_directory))

        self._root_dir = root_directory

        # Dictionary of children directories
        self._children = {}

        self._children["root"] = root_directory

    def get_validation_name(self, name):
        dir_name = os.path.basename(os.path.normpath(name))
        return "{0}{1}validate_{2}".format(self._children["root"], os.path.sep, dir_name)

    def get_next_name(self, key, name):
        idxs = [0]
        parent_folder = self._children[key]
        dirs = os.listdir(parent_folder)

        last_word = name.split("_")[-1]
        for directory_name in dirs:
            if last_word in directory_name:
                idx = int(directory_name.split("_")[-1].strip(last_word))
                idxs.append(idx)
        next_name = name + str(max(idxs) + 1)
        result_name = os.path.join(parent_folder, next_name)

        if os.path.exists(result_name):
            self._logger.error("Next director exists....something went wrong with calculating new directory name")
            exit(0)

        return result_name

    def create(self, key, path):
        if not os.path.exists(path):
            os.mkdir(path)

        self._children[key] = path
        self.log("{0} - added the path {1} with key {2}".format(self.name, path, id))

    def log(self, msg):
        if self._logger is not None:
            self._logger.trace(msg)

    def __setitem__(self, key, value):
        self.create(key, value)

    def __getitem__(self, item):
        return self._children[item]

    def remove(self, key):
        if key not in self._children.keys():
            self.log("{0} - key not present so cannot be removed".format(self.name))
        # TODO - best practice?
        del self._children[key]

    def clear(self):
        self._children.clear()
        self.log("{0} - cleared all directories".format(self.name))


if __name__ == '__main__':
    rospy.init_node("ga_pid_optimization")

    parser = argparse.ArgumentParser(description="Run a Ziegler-Nichols optimisation.")
    parser.add_argument(
        "--validate", type=str, nargs="+", default="", help="Validate a learning run by removing exploration"
    )

    parser.add_argument(
        "--repeat", type=int, default=1, help="Number of times to repeat the learning process"
    )

    parser.add_argument(
        "--root", type=str, default="runs", help="Root directory name"
    )

    args = parser.parse_args()
    
    logger = SimpleLogger()
    dir_manager = DirectoryManager(logger, args.root)
    
    ros_env = ROSBehaviourInterface()

    for prop_value in PROP_VALUES:
        result_dir = dir_manager.get_next_name("root", "ziegler_pid")
        
        filename = os.path.basename(sys.argv[0])
        dir_manager["run"] = result_dir
        os.system("cp {0} {1}".format(filename, result_dir))
        # os.system("cp /home/gordon/rosbuild_ws/ros_simple_rl/src/ga_optimization/ga_optimizer.py {0}".format(
        #     indexed_results_dir))
        os.system("cp /home/gordon/rosbuild_ws/ros_simple_rl/src/ga_optimization/processes/pilot_pid_process.py {0}".format(
            result_dir))
        # os.system("cp /home/gordon/rosbuild_ws/ros_simple_rl/src/ga_optimization/ga.py {0}".format(
        #     indexed_results_dir))

        process = PilotPidProcess(ros_env, result_dir)
        
        # [proportional, derivitive, integral]
        gains = [prop_value, 0, 0]
        np.savetxt("{0}/gains.csv".format(result_dir), np.array(gains), delimiter=",")

        process.get_response(0, gains)
        
