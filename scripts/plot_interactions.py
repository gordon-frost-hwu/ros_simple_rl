#! /usr/bin/python
import pdb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

print("Using backend: {0}".format(plt.get_backend()))
import math
import sys
from copy import deepcopy
# Imports that are used for the ordering of the input files according to their index number
from collections import OrderedDict
from itertools import cycle
import re
from matplotlib import rc
rc('text', usetex=True)

line_cycle = cycle(["-",":","--","-.",])
marker_cycle = cycle(["p","v","o","D","s",])
plt.ion()
DEFAULT_PATH = '/tmp/learning_interactions.txt'

def consolidateData(paths_to_avg_data_over):
    data = None
    # TODO: Fix bug occuring when only 1 results file is presented and -m option given!!!!
    num_files = len(paths_to_avg_data_over)
    print("DEBUG: length of paths: {0}".format(num_files))
    min_num_samples = 100000
    _idx = 0
    for path in paths_to_avg_data_over:
        d = np.loadtxt(path, comments='#', delimiter=DELIMITER)
        if data is None:
            data = np.zeros(d.shape)
            data_statistics = np.zeros([d.shape[0], num_files])
        if d.shape[0] < min_num_samples:
            min_num_samples = d.shape[0]
            data = np.delete(data, range(min_num_samples, data.shape[0]), 0)
            data_statistics = np.delete(data_statistics, range(min_num_samples, data_statistics.shape[0]), 0)
        data[0:min_num_samples] += d[0:min_num_samples]
        data_statistics[:, _idx] = d[0:min_num_samples, INDEX_TO_PLOT]
        _idx += 1
    data /= len(paths_to_avg_data_over)
    print("DATA STATS: {0}".format(data_statistics.shape))

    return data_statistics

def sort_input(input_list, num_runs):
    fake_idx = 0
    results_dict = {}
    for filepath in input_list:
        try:
            split_filepath = filepath.split("/")
            res_dir_name, filename = split_filepath[-2], split_filepath[-1]
        except IndexError:
            filename = filepath
        print(filename)
        try:
            result_number = re.findall(r"[-+]?\d*\.*\d+", filename)[-1]
        except:
            print("No index found in filename----trying DIR name")
            try:
                result_number = re.findall(r"[-+]?\d*\.*\d+", res_dir_name)[-1]
            except:
                print("No Index found in DIR name---assigning FAKE one")
                result_number = str(fake_idx)
                fake_idx += 1
        if "north" in filename:
            tag = "north"
        elif "east" in filename:
            tag = "east"
        elif "yaw" in filename:
            tag = "yaw"
        else:
            tag = None
        results_dict[result_number, tag] = filepath
    return results_dict

# Depending on script arguments, either use the supplied path as argument or use default
args = sys.argv
args.pop(0)
print("Args before option parsing: {0}".format(args))

if "--raw" in args:
    # Just plot all argument paths - only the step function will be available
    RAW_PLOT = True
    args.remove("--raw")
else:
    RAW_PLOT = False

if "-i" in args:
    # Telling which column to plot (from the end one, -1)
    INDEX_TO_PLOT = int(args[args.index("-i") + 1])
    args.remove(args[args.index("-i") + 1])
    args.remove(args[args.index("-i")])
else:
    INDEX_TO_PLOT = -1

if "-d" in args:
    DELIMITER = args[args.index("-d") + 1]
    args.remove(args[args.index("-d") + 1])
    args.remove("-d")
else:
    DELIMITER = "\t"

if "--axis" in args:
    AXIS_LABELS = True
    x_label = args[args.index("--axis") + 1]
    y_label = args[args.index("--axis") + 2]
    args.remove("--axis")
    args.remove(x_label)
    args.remove(y_label)
    print("preprocessing axis label strings ...")
    x_label = x_label.replace("_", " ")
    y_label = y_label.replace("_", " ")
else:
    AXIS_LABELS = False
if "--legend" in args:
    LEGEND_GIVEN = True
    print("WARNING!!!!")
    print("WARNING: Legend argument must be last argument given...")
    print("WARNING!!!!")
    opt_indx = args.index("--legend")
    legend_names = args[opt_indx+1:]
    args.remove("--legend")
    for n in legend_names:
        args.remove(n)
else:
    LEGEND_GIVEN = False

if "--ylim" in args:
    Y_LIM = True
    opt_indx = args.index("--ylim")
    y_lim_lower = int(args[opt_indx+1])
    y_lim_upper = int(args[opt_indx+2])
    print(y_lim_lower)
    args.remove("--ylim")
    args.remove(str(y_lim_lower))
    args.remove(str(y_lim_upper))
else:
    Y_LIM = False
if "--nomarker" in args:
    NOMARKER = True
    args.remove("--nomarker")
else:
    NOMARKER = False

if "-o" in args:
    SAVE_TO_FILENAME = args[args.index("-o") + 1]
    args.remove(args[args.index("-o") + 1])
    args.remove("-o")
    print("WARNING--- -o option only saves to file when -m option is also given....")
else:
    SAVE_TO_FILENAME = None

if not RAW_PLOT:
    if "-r" in args:
        num_result_paths = int(args[args.index("-r") + 1])
        args.remove(args[args.index("-r") + 1])
        args.remove("-r")
    else:
        print("MUST supply number of results paths to script using -r flag...")
        exit(0)

    paths_ = args[0:num_result_paths]
    for _p in paths_:
        args.remove(_p)
    print("_DEBUG: paths_: {0}".format(paths_))
    print("_DEBUG: Args before option parsing: {0}".format(args))

    # -------
    # AT THIS STAGE WE HAVE: paths_ & list of operations to do in args
    # -------

    if "--sample" in args:
        SAMPLE_AT_STEP_NUMBER = int(args[args.index("--sample") + 1])
        args.remove(args[args.index("--sample") + 1])
        args.remove(args[args.index("--sample")])
    else:
        SAMPLE_AT_STEP_NUMBER = False

    if "--ordered" in args:
        ARGS_SORTED = True
        args.remove("--ordered")
    else:
        ARGS_SORTED = False

    if not ARGS_SORTED:
        sorted_paths = sort_input(paths_, 0)
    else:
        sorted_paths = {}
        idx = 0
        for path in paths_:
            sorted_paths[(str(idx), None)] = path
            idx += 1
    print("sorted_paths: {0}".format(sorted_paths))
    # Plot all of the desired paths that were given as input to the script using their indices from the dict
    idxs = [int(k[0]) for k in sorted_paths.viewkeys()] # get a list of the integer idx values given to script
    idxs = list(OrderedDict.fromkeys(idxs))
    idxs.sort()
    if len(sorted_paths.values()) < len(paths_):
        print("---------------")
        print("WARNING---integer keys for filenames clash. Plotting randomly")
        print("WARNING---constructing sorted_paths randomly")
        sorted_paths = {}
        p_idx = 0
        idxs = []
        for _path in paths_:
            sorted_paths[(str(p_idx), None)] = _path
            idxs.append(p_idx)
            p_idx += 1
        print("--------------")
    else:
        pass

    print("sorted_paths: {0}".format(sorted_paths))
    print("idxs: {0}".format(idxs))

    print("args: {0}".format(args))
    if len(args) == 0:
        args.append("plot_interactions.py")
    fig, ax = plt.subplots()


    plt.plot([0 for x in range(5)], "w", alpha=0.0)
    structured_data = consolidateData([sorted_paths[str(_idx), None] for _idx in idxs])
    print("DEBUG: consolidatedData: {0}".format(structured_data.shape))

    # ------
    # Step through the operations applying them sequentially.
    # Need the data in an appropriate structure beforehand!
    # ------
    for operation in args:
        mean_structured_data = structured_data.sum(axis=1) / num_result_paths
        if operation == "-m":
            # average runs from the multiple results file paths
            print("Plotting Average")
            print(mean_structured_data)
            ax.plot(mean_structured_data)
            print("DEBUG: mean_structured_data.shape: {0}".format(mean_structured_data.shape))
            print("DEBUG: SAMPLE_AT_STEP_NUMBER: {0}".format(SAMPLE_AT_STEP_NUMBER))

            if SAVE_TO_FILENAME is not None:
                to_save = np.ones([max(mean_structured_data.shape), 2])
                to_save[:,0] = range(0, max(mean_structured_data.shape))
                to_save[:, 1] = mean_structured_data[:]
                np.savetxt(SAVE_TO_FILENAME, to_save, delimiter="\t")

            if SAMPLE_AT_STEP_NUMBER:
                print("(SCALED by 3.14) Absolute error at sample point {0}: {1}".format(SAMPLE_AT_STEP_NUMBER,
                                                                                   3.14 * mean_structured_data[SAMPLE_AT_STEP_NUMBER]))
                raw_of_sample = mean_structured_data[SAMPLE_AT_STEP_NUMBER:-1]
                mean_of_sample = sum(abs(raw_of_sample)) / len(raw_of_sample)
                print("(     RAW      ) Mean Absolute Error from sample point {0} to {1}: {2}".format(SAMPLE_AT_STEP_NUMBER,
                                                                                                 max(mean_structured_data.shape),
                                                                                                 mean_of_sample))

        if operation == "--mse":
            structured_data = structured_data**2

        if operation == "--integrate":
            integral_of_dim = []
            for file_idx in range(structured_data.shape[1]):
                integral_of_dim.append(sum(abs(structured_data[:, file_idx])))
            ax.plot(integral_of_dim)

        if operation == "--variance":
            print("STRUCTURED_DATA: {0}".format(structured_data.shape))
            print("MEAN_STRUCTURED_DATA: {0}".format(mean_structured_data.shape))
            cont_variable = structured_data[:,:]    # should be size: [Num_samples x Num_files]
            cont_variable_mean = np.array([mean_structured_data.tolist(),]*min(structured_data.shape)).transpose() # should be size: [Num_samples x Num_files]
            print("cont_variable: {0}".format(cont_variable.shape))
            print("cont_variable_mean: {0}".format(cont_variable_mean.shape))
            variance =  cont_variable - cont_variable_mean    # should be size: [Num_steps x Num_files]
            print("SUBTRACT: {0}".format(variance.shape))
            variance = variance**2
            print("SQUARE: {0}".format(variance[0:4,:]))
            variance = variance.sum(axis=1) / (num_result_paths - 1) # / n-1 for Bessel's correction
            print("STD STATS: {0}".format(variance.shape))
            print("VARIANCE: {0}".format(variance[0:4]))

            try:
                std_deviation = np.sqrt(variance)
            except:
                print("Runtime error in STD Deviation calculation")
                print(variance)
                print("Negative values used for sqrt(variance)")
                exit(0)

            x = range(0, max(mean_structured_data.shape))
            ax.plot(mean_structured_data, color="b", linestyle="-")
            plt.fill_between(x, mean_structured_data-std_deviation, mean_structured_data+std_deviation, color='b', linestyle="--",
                                     linewidth=0.0, alpha=0.2)
            ax.plot(x, mean_structured_data - std_deviation, "b", linestyle="--")
            ax.plot(x, mean_structured_data + std_deviation, "b", linestyle="--")

        if operation == "--diff":
            # differentiate, i.e. show the change of the variable
            # differentiate each files variable
            diffs = np.zeros(structured_data.shape)
            for file_idx in range(structured_data.shape[1]):
                col = structured_data[:, file_idx]
                ax.plot(col)
                # normalise the data
                col /= abs(min(col))
                print(col)
                diff = []
                prev_sample = 0.0
                for sample in col:
                    diff.append(sample-prev_sample)
                    prev_sample = deepcopy(sample)
                diffs[:,file_idx] = diff
            # take the average of the differentiated variables
            if num_result_paths > 1:
                average_diff = diffs.sum(axis=1) / (num_result_paths - 1)
            elif num_result_paths == 1:
                average_diff = diffs.sum(axis=1) / (num_result_paths)
            ax.plot(average_diff*1000)

            # apply a sliding/moving average to the differiented average
            from utilities.moving_differentiator import SlidingWindow
            WINDOW_SIZE = 10
            sliding_window = SlidingWindow(window_size=WINDOW_SIZE)
            converged_indexes =[]
            for index, sample in enumerate(average_diff):
                window = sliding_window.getWindow(sample)
                print("sample: {0}".format(window))
                moving_avg = sum(window) / float(WINDOW_SIZE)
                print(moving_avg)
                # if index > WINDOW_SIZE:
                if abs(moving_avg) < 0.001:
                    ax.plot(index, sample, "rx")
                    converged_indexes.append(index)
            try:
                print("Converged indexes: {0}".format(converged_indexes[0:5]))
            except:
                pass
    
    # if len(plt.axes().lines) == 0:
    #     print("No OPERATORS given so plotting the plain files")
    #     plotted_objects = []
    #     # no mean or mse or integral desired to plot so just plot the raw data
    #     from random import randint
    #     _idx = 0
    #     for column_idx in range(structured_data.shape[1]):
    #         if not NOMARKER:
    #             line, = ax.plot(structured_data[:, column_idx], marker=marker_cycle.next(), markevery=randint(30, 100), markersize=10)
    #         else:
    #             line, = ax.plot(np.linspace(-1.0, 1.0, 100), structured_data[:, column_idx])
    #         plotted_objects.append(line)
    #         if "-s" in args:
    #             print("File path plotted: {0}".format(sorted_paths[str(idxs[_idx]), None]))
    #             plt.draw()
    #             unused_variabResle = raw_input("press any key to continue ... ")
    #             _idx += 1
else:
    # Plot the raw file data
    print("raw args: {0}".format(args))
    paths_ = []
    for arg in args:
        if "-" not in arg:
            paths_.append(arg)
            #args.remove(arg)
    print("args after parsing")
    print(args)
    print(paths_)

    # -------
    # Sort the files sequentially
    sorted_paths = sort_input(paths_, 0)
    print("sorted_paths: {0}".format(sorted_paths))
    # Plot all of the desired paths that were given as input to the script using their indices from the dict
    idxs = [int(k[0]) for k in sorted_paths.viewkeys()] # get a list of the integer idx values given to script
    idxs = list(OrderedDict.fromkeys(idxs))
    idxs.sort()
    if len(sorted_paths.values()) < len(paths_):
        print("---------------")
        print("WARNING---integer keys for filenames clash. Plotting randomly")
        print("WARNING---constructing sorted_paths randomly")
        sorted_paths = {}
        p_idx = 0
        idxs = []
        for _path in paths_:
            sorted_paths[(str(p_idx), None)] = _path
            idxs.append(p_idx)
            p_idx += 1
        print("--------------")
    else:
        pass

    paths = [sorted_paths[str(_idx), None] for _idx in idxs]
    # -------

    fig, ax = plt.subplots()

    for file in paths:
        data = np.loadtxt(file, delimiter=DELIMITER, comments="#")
        try:
            ax.plot(data[:,INDEX_TO_PLOT])
        except IndexError:
            print("WARNING WARNING WARNING WARNING ---- INDEX ERROR, PLOTTING -1 COLUMN INSTEAD!!!!")
            ax.plot(data[:,-1])

        if "-s" in args:
            print("plotting: {0}".format(file))
            plt.draw()
            unused_variable = raw_input("press any key to continue ... ")

if LEGEND_GIVEN:
    # plt.legend() options: ncol, nrow, loc, bbox_to_anchor
    legend_names = [r"\textbf{    }".replace("    ", name) for name in legend_names]
    plt.legend(plotted_objects, legend_names, loc="best", prop={'size':30})#, bbox_to_anchor=(0.0, 1.0), ncol=len(legend_names))

if AXIS_LABELS:
    plt.xlabel(r"\textbf{    }".replace("    ", x_label), fontsize=30)
    plt.ylabel(r"\textbf{    }".replace("    ", y_label), fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=30)

if Y_LIM == True:
    plt.ylim([y_lim_lower, y_lim_upper])
    plt.draw()

unused_variable = raw_input("press any key to continue ... ")
exit(0)


# ------------------------
# OLD CODE
# ------------------------
if "-b" in args:
    # run the avg over batches of the sorted input paths
    BATCH = True
    batch_amount = int(args[args.index("-b") + 1])
    args.remove(args[args.index("-b") + 1])
    args.remove("-b")
else:
    BATCH = False

if "--batch" in args:
    assert SAVE_TO_FILENAME is not None, "External script calling this script --- must supply: -o filename.csv"
    SAVE_TO_FILE_ONLY = True
    args.remove("--batch")
else:
    SAVE_TO_FILE_ONLY = False
if "--integrate" in args:
    INTEGRATE_DIM = True
    args.remove("--integrate")
else:
    INTEGRATE_DIM = False
if "--sample" in args:
    SAMPLE_AT_STEP_NUMBER = int(args[args.index("--sample") + 1])
    args.remove(args[args.index("--sample") + 1])
    args.remove(args[args.index("--sample")])
else:
    SAMPLE_AT_STEP_NUMBER = None
if "--mse" in args:
    CALC_MSE = True
    args.remove(args[args.index("--mse")])
else:
    CALC_MSE = False
if "-n" in args:
    # normalise the reward by the use of a file which shows the number of steps per episode
    NORMALISE = True
    normalising_file_data = np.loadtxt(args[args.index("-n") + 1])
    args.remove(args[args.index("-n") + 1])
    args.remove("-n")
else:
    NORMALISE = False
if "-s" in args:
    STEP = True
    args.remove("-s")
else:
    STEP = False
if "-a" in args:
    SHOW_ALL_COLUMNS = True
    args.remove("-a")
else:
    SHOW_ALL_COLUMNS = False

if "--variance" in args:
    SHOW_VARIANCE = True
    args.remove("--variance")
else:
    SHOW_VARIANCE = False

if len(args) > 1:
    print("Using paths: {0}".format(args[1:]))
    paths = args[1:]
else:
    paths = DEFAULT_PATH

print("Args after option parsing: {0}".format(args))

if SAVE_TO_FILE_ONLY and CALC_MSE:
    assert SAMPLE_AT_STEP_NUMBER is not None, "batch external script calling this one --- must give a --sample argument"



if SHOW_ALL_COLUMNS:
    from utilities.logged_data_to_dict import transform_log_to_dict
    for path in paths:
        d = np.loadtxt(path, comments='#', delimiter=DELIMITER)
        try:
            d_dict = transform_log_to_dict(path, delimiter="\t")
        except:
            d_dict = None
        # print("Data Shape: {0}".format(d.shape))
        desired_num_cols = 2.0
        desired_num_rows = math.ceil(float(d.shape[1]) / desired_num_cols)
        fig, axs = plt.subplots(int(desired_num_rows), int(desired_num_cols))
        print(axs)

        fig.suptitle(path)
        if d_dict is not None:
            results_keys = d_dict.keys()
        idx = 0
        for axs_row in axs:
            for ax_row_col in axs_row:
                print(idx)
                if d_dict is None:
                    # print(d.shape[1])
                    if not idx > d.shape[1] - 1:
                        ax_row_col.plot(d[:, idx])
                        ax_row_col.set_title("Idx {0}".format(idx))
                else:
                    ax_row_col.plot(d_dict[results_keys[idx]])
                    ax_row_col.set_title("{0}".format(results_keys[idx]))
                idx += 1

def averageData(paths_to_avg_data_over):
    data = None
    # TODO: Fix bug occuring when only 1 results file is presented and -m option given!!!!
    num_files = len(paths_to_avg_data_over)
    print("DEBUG: length of paths: {0}".format(num_files))
    min_num_samples = 100000
    _idx = 0
    for path in paths_to_avg_data_over:
        d = np.loadtxt(path, comments='#', delimiter=DELIMITER)
        if data is None:
            data = np.zeros(d.shape)
            data_statistics = np.zeros([d.shape[0], num_files])
        if d.shape[0] < min_num_samples:
            min_num_samples = d.shape[0]
            data = np.delete(data, range(min_num_samples, data.shape[0]), 0)
            data_statistics = np.delete(data_statistics, range(min_num_samples, data_statistics.shape[0]), 0)
        data[0:min_num_samples] += d[0:min_num_samples]
        data_statistics[:, _idx] = d[0:min_num_samples, INDEX_TO_PLOT]
        _idx += 1
    data /= len(paths_to_avg_data_over)
    print("DATA STATS: {0}".format(data_statistics.shape))

    rewards = data[:,INDEX_TO_PLOT]
    variance = data_statistics - rewards[:, np.newaxis]
    print("STD STATS: {0}".format(variance.shape))
    variance = variance**2
    print("STD STATS: {0}".format(variance.shape))
    variance = variance.sum(axis=1) / num_files - 1 # / n-1 for Bessel's correction
    print("STD STATS: {0}".format(variance.shape))
    print(variance)

    try:
        std_deviations = np.sqrt(variance)
    except:
        print("Runtime error in STD Deviation calculation")

    if NORMALISE:
        rewards = rewards / normalising_file_data
    if SAVE_TO_FILENAME is not None:
        np.savetxt(SAVE_TO_FILENAME, data, delimiter="\t")

    return rewards, std_deviations, variance

if not SHOW_ALL_COLUMNS:
    plt.figure(1)
    if AVERAGE:
        if BATCH:
            # batches = np.array(np.array(range(len(idxs) / batch_amount)) * batch_amount)
            batches = deepcopy(idxs)

            # Check if we need to balance the number of idxs in order to reshape according to batch size
            print("Number of batches: {0}".format(len(batches)))
            if (len(batches) % batch_amount) != 0:
                extend_by = batch_amount - (len(batches) % batch_amount)
            else:
                extend_by = 0
            print("extend by: {0}".format(extend_by))
            if extend_by != 0:
                batches.extend([1 for _ in range(extend_by)])
            batches = np.array(batches).reshape([-1, batch_amount])
            print(batches)
            for batch in batches:
                avg_for_batch = averageData([sorted_paths[str(_idx), None] for _idx in batch])
                plt.plot(avg_for_batch)
                useless = raw_input("Pressn ENTER to continue ...")
        else:
            averaged_data, std_deviation, variance = averageData(paths)
            x = range(0, len(averaged_data))
            plt.plot(x, averaged_data)
            print("std_deviation: {0}".format(std_deviation))
            if SHOW_VARIANCE:
                # Documention for fill_between: http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.fill_between
                plt.fill_between(x, averaged_data-std_deviation, averaged_data+std_deviation, color='b', linestyle="--",
                                 linewidth=0.0, alpha=0.2)
                plt.plot(x, averaged_data-std_deviation, "b", linestyle="--")
                plt.plot(x, averaged_data+std_deviation, "b", linestyle="--")
            # plt.errorbar(averaged_data, yerr=variance)

        # plt.scatter(range(0, rewards.shape[0]), rewards, cmap=np.arange(100))
        # plt.plot(num_steps, 'g')
        # plt.legend(["reward", "num steps"])

        #plt.figure(2)
        #plt.plot(avg_reward_per_episode)
    else:
        # plot all of the results on the same plot
        plt.figure(1)
        integral_of_dim = []
        point_analysis_points = []
        print("idxs: {0}".format(idxs))
        for idx in idxs:
            print("loop idx: {0}".format(idx))
            path = sorted_paths[str(idx), None]
            data = np.loadtxt(path, comments='#', delimiter=DELIMITER)

            try:
                rewards = data[:,INDEX_TO_PLOT]
            except IndexError:
                print("Failed to read data from path: {0}".format(path))
                exit(0)

            if NORMALISE:
                rewards = rewards / normalising_file_data

            if SAMPLE_AT_STEP_NUMBER is not None:
                print(type(rewards))
                point_analysis_points.append(rewards[SAMPLE_AT_STEP_NUMBER])

            # plot the different variables (held in different columns)
            if not INTEGRATE_DIM and not CALC_MSE:
                to_plot = rewards
                plt.plot(to_plot)
            elif INTEGRATE_DIM:
                integral_of_dim.append(sum(abs(rewards)))
                to_plot = integral_of_dim
            elif CALC_MSE:
                to_plot = rewards**2
                plt.plot(to_plot)
            # plt.scatter(range(0, rewards.shape[0]), rewards, cmap=np.arange(100))
            #plt.plot(num_steps, 'g')
            print(point_analysis_points)
            if STEP:
                print("Just Plotted: {0}".format(path))
                unused_variable = raw_input("press any key to continue ... ")
        # to_plot = integral_of_dim
        # plt.plot(to_plot)
        if SAVE_TO_FILENAME is not None:
            to_save = np.ones([len(to_plot), 2])
            to_save[:,0] = range(0, len(to_plot))
            to_save[:, 1] = to_plot
            np.savetxt(SAVE_TO_FILENAME, to_save, delimiter="\t")

#from utilities.moving_differentiator import DifferentiateVariable
#plt.figure(3)
#window_sizes = [2, 4, 8, 16, 32]
#for window_size in window_sizes:
#    rewards_per_episode_gradient = DifferentiateVariable(window_size=window_size)
#
#    gradients = []
#    for reward in rewards:
#        gradients.append(rewards_per_episode_gradient.gradient(reward))
#
#    plt.plot(gradients)
#plt.legend([str(window) for window in window_sizes])

# plt.savefig("/tmp/test_fig.eps", dpi=4)
if not SAVE_TO_FILE_ONLY:
    plt.show()
    input("Press any key ...")
