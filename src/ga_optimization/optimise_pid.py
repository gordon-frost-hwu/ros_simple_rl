#! /usr/bin/python
import roslib
roslib.load_manifest("ros_simple_rl")

import os
import rospy
import sys
from ga_optimizer import GAOptimizer
from processes.pilot_pid_process import PilotPidProcess

CONFIG = {
    "num_repetitions": 9,
    "num_generations": 30,
    "sol_per_pop": 8,   # was 8
    "num_parents_mating": 4
}
# Short test setup
# CONFIG = {
#     "num_repetitions": 5,
#     "num_generations": 2,
#     "run_time": 15,
#     "sol_per_pop": 4,   # was 8
#     "num_parents_mating": 2
# }

if __name__ == '__main__':
    rospy.init_node("ga_pid_optimization")

    args = sys.argv
    if "-r" in args:
        results_dir_prefix = args[args.index("-r") + 1]
    else:
        results_dir_prefix = "ga_pid_tuning"

    for run in range(CONFIG["num_repetitions"]):
        indexed_results_dir = "/home/gordon/data/tmp/{0}{1}".format(results_dir_prefix, run)

        if not os.path.exists(indexed_results_dir):
            os.makedirs(indexed_results_dir)
        filename = os.path.basename(sys.argv[0])
        os.system("cp {0} {1}".format(filename, indexed_results_dir))
        os.system("cp /home/gordon/rosbuild_ws/ros_simple_rl/src/ga_optimization/ga_optimizer.py {0}".format(
            indexed_results_dir))
        os.system("cp /home/gordon/rosbuild_ws/ros_simple_rl/src/ga_optimization/processes/pilot_pid_process.py {0}".format(
            indexed_results_dir))
        os.system("cp /home/gordon/rosbuild_ws/ros_simple_rl/src/ga_optimization/ga.py {0}".format(
            indexed_results_dir))

        process = PilotPidProcess(indexed_results_dir)
        pilot = GAOptimizer(CONFIG, indexed_results_dir, process)
        pilot.run()
