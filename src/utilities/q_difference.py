#! /usr/bin/env python
import random
import copy
import pickle
import numpy as np
import sys

import matplotlib.pyplot as plt

# following number (256) is from boundary conditions as in the total state-action space, some actions
# around the boundary are illegal (4 possible actions at 4 possible states per "point" where there are 16 = 256
# also 16 sa illegal pairs due to the goal state never "being visited"
#NUM_ILLEGAL_SA_PAIRS = 256 + 16     # For old 4x4 grid with 4 headings per point/grid location
NUM_ILLEGAL_SA_PAIRS = 44
MOVING_AVG_SIZE = 15

class Data(object):
    def __init__(self):
        self.qtables = []
        self.ntables = []
        self.qtables_length = None
        self.ntables_length = None
        self.q_variance = None
        self.num_s_a_visited = []
        self.cut_num_episodes = 0
        self.q_offset = 0
        self.n_offset = 0
        self.percentage_visited = 0
        self.tmp_count = 0
        self.q_moving_avg = []

    def loadTables(self, directory, OLD_DATA=False):
        print("Results of: {0}".format(directory))
        if not OLD_DATA:
            q_table_file_path = directory + "q_tables.txt"
            q_table_file = open(q_table_file_path, 'r')
            q_tables_encrypted = q_table_file.read().split(",")
            for table in q_tables_encrypted:
                if len(table) != 0:
                    self.qtables.append(pickle.loads(table))

            #print("Last Table is: {0}".format(self.qtables[-1]))

            n_table_file_path = directory + "n_tables.txt"
            n_table_file = open(n_table_file_path, 'r')
            n_tables_encrypted = n_table_file.read().split(",")
            for table in n_tables_encrypted:
                if len(table) != 0:
                    self.ntables.append(pickle.loads(table))

        else:
            # OLD METHOD before conversion to text
            q_table_file_path = directory + "q_tables"
            with open(q_table_file_path, 'rb') as f:
                qtables = pickle.load(f)
                f.close()

            n_table_file_path = directory + "n_tables"
            with open(n_table_file_path, 'rb') as f:
                ntables = pickle.load(f)
                f.close()
        
        self.qtables_length = len(self.qtables)
        self.ntables_length = len(self.ntables)

    def pruneTables(self, desired_num_episodes):
        number_to_prune = self.qtables_length - desired_num_episodes
        if desired_num_episodes != self.qtables_length:
            print("     Pruning the tables to length: %s" % desired_num_episodes)
            for i in range(number_to_prune):
                self.qtables.pop()
                self.ntables.pop()
        else:
            print("     Requested number of episodes is more than data set, proceeding!")
        self.qtables_length = len(self.qtables)
        self.ntables_length = len(self.ntables)

    def multiplyZero(self, table, table_prev):
        difference_array = table*table_prev
        difference = np.linalg.norm(difference_array)
        return difference

    def normDifference(self, table, table_prev):
        """ Input Qtables of type lists, returns the numeric difference """
        # First convert list to np array
        norm_table = np.linalg.norm(table)
        norm_table_prev = np.linalg.norm(table_prev)
        difference = abs(norm_table - norm_table_prev)
        #difference_array = table - table_prev
        #difference = np.linalg.norm(difference_array)
        return difference

    def analyseQ(self):
        def compare(x, i):
            #print("x: %s, i: %s" % (x, i)
            curr_q = x
            prev_q = self.qtables[i-1]
            Qtable_array = np.array(curr_q)
            Qtable_prev_array = np.array(prev_q)

            val = self.normDifference(Qtable_array, Qtable_prev_array)
            self.q_moving_avg.append(copy.deepcopy(val))
            if len(self.q_moving_avg) == MOVING_AVG_SIZE:
                if self.q_moving_avg != [0 for i in range(MOVING_AVG_SIZE)]:
                    for i in range(self.q_moving_avg.count(0)):
                        self.q_moving_avg.remove(0)
                np_array = np.array(self.q_moving_avg)
                moving_avg = np_array.sum()/len(np_array)
                if moving_avg < 0.15:
                    print("Comparison Num: %s; val: %s" % (self.tmp_count, np_array))
                self.q_moving_avg = []
            self.tmp_count += 1
            value = val + self.q_offset
            self.q_offset = copy.copy(value)
            return value

        # Construct the list of difference values following
        self.q_variance = [compare(x,i) for i, x in enumerate(self.qtables)][1:]

    def analyseN(self):
        def compare_n(x, i):
            total_unobserved_count = 0
            for row in x:
                num_zeros_in_row = row.count(1)
                total_unobserved_count += num_zeros_in_row
            if total_unobserved_count != 0:
                s_a_space_size = (len(x)*len(x[0]))-NUM_ILLEGAL_SA_PAIRS
            num_visited = s_a_space_size-(total_unobserved_count- NUM_ILLEGAL_SA_PAIRS)
            #print("Explored %s State-Action pairs out of %s" % (num_visited, s_a_space_size)

            self.num_s_a_visited.append(num_visited)
            curr_q = x
            prev_q = self.ntables[i-1]
            Qtable_array = np.array(curr_q)
            Qtable_prev_array = np.array(prev_q)

            val = self.normDifference(Qtable_array, Qtable_prev_array)
            value = val + self.n_offset
            #print("val: %s, offset_n: %s, gives value: %s" % (val, offset_n, value)
            self.n_offset = copy.copy(value)
            return value

        n_variance = [compare_n(x,i) for i, x in enumerate(self.ntables)][1:]

        # Can pick any table as they do not change shape
        s_a_space_size = (len(self.ntables[0])*len(self.ntables[0][0])) - NUM_ILLEGAL_SA_PAIRS
        total_unobserved_count = 0
        for row in self.ntables[-1]:
            num_zeros_in_row = row.count(1)
            total_unobserved_count += num_zeros_in_row
            num_visited = s_a_space_size-(total_unobserved_count-NUM_ILLEGAL_SA_PAIRS)


        tmp_array = np.array(self.num_s_a_visited)
        self.percentage_visited = (tmp_array / float(s_a_space_size)).tolist()
        #print("Percentage Visited List: "
        #print self.percentage_visited
        print("     Explored %s State-Action pairs out of %s = %.4s percent" % (num_visited, s_a_space_size, self.percentage_visited[-1]))

    def plotData(self, colour, xlim):
        plt.figure()
        #plt.title("Number of State-Action pairs Visited")
        plt.ylabel('Percentage Visited (%)', fontsize='x-large')
        plt.xlabel('Episodes', fontsize='x-large')
        plt.plot(self.percentage_visited, colour, marker=None, linestyle="-")
        plt.ylim([0, 1])    # set limits for x and y axes of the plot
        plt.xlim([0, xlim])
        plt.grid()


        # Plot the figures
        plt.figure()
        #plt.title("Relative Norm in Q")
        plt.ylabel('Q-table Difference (Norm of $Q_e$ - $Q_{e-1}$)', fontsize='x-large')
        plt.xlabel('Episodes', fontsize='x-large')
        plt.plot(self.q_variance, colour, marker=None, linestyle="-")
        #plt.ylim([0, 200])    # set limits for x and y axes of the plot
        plt.xlim([0, xlim])
        plt.grid()

    def saveasCSV(self, input_path, output_path):
        """ input_path: variable used just to get the name of the results directory
            output_path: this is where the csv files will be saved
        """
        percent = ','.join(map(str, self.percentage_visited))
        q_diff = ','.join(map(str, self.q_variance))
        name = input_path.split("/")
        print("Split name: %s" % name)
        name = name[-2]

        with open(output_path+name+"_percentage_visited.csv", "w") as f:
            f.write(str(percent))
            f.close()

        with open(output_path+name+"_q_difference.csv", "w") as f:
            f.write(str(q_diff))
            f.close()

if __name__ == '__main__':
    # ra_policy_dir = "/home/gordon/Documents/Dropbox/Conference_Papers/IROS_2014/results_for_paper/481e_ra/"
    # lv_policy_dir = "/home/gordon/Documents/Dropbox/Conference_Papers/IROS_2014/results_for_paper/480e_lv/"
    # input_data = [[ra_policy_dir, "r-"], [lv_policy_dir, "b-"]]
    # input_data = [["/home/gordon/m_files_octave/qtable-approximation/training_data/large_dataset_6/", "r-"]]
    # input_data = [["/home/gordon/Documents/Dropbox/Conference_Papers/tabular_q_emotion_results/", "r-"]]

    # common_file_path = "/home/gordon/Documents/Dropbox/Conference_Papers/OCEANS_2014/oceans_results/"
    # filepath_0 = common_file_path +"p0_gamma/"
    # filepath_1 = common_file_path +"p2_gamma/"
    # filepath_2 =  common_file_path +"p4_gamma/"
    # filepath_3 =  common_file_path +"p6_gamma/"
    # filepath_4 =  common_file_path +"p8_gamma/"
    # filepath_5 =  common_file_path +"p10_gamma/"

    args = sys.argv
    if len(args) == 1:
        show_results = True
        input_filepath = "/home/gordon/Dropbox/Conference_Papers/emotion_results/tmp_tabular_q_emotion_results/"
        output_filepath = "/home/gordon/Dropbox/Conference_Papers/emotion_results/tmp_tabular_q_emotion_results/post_processing/"
    else:
        input_filepath = args[1]
        output_filepath = args[2]
        show_results = bool(int(args[-1]))

    common_file_path = "/home/gordon/Dropbox/Conference_Papers/emotion_results/"
    filepath_0 = common_file_path +"tabular_q_emotion_results/"
    filepath_1 = common_file_path +"tmp_tabular_q_emotion_results/"
    #input_data = [[filepath_0, "r-"], [filepath_1, "g-"]]
    input_data = [[input_filepath, output_filepath, "g-"]]

    # Configuration Parameters
    XLIMIT = 20

    # ----------------------------------------
    # For trying different methods of calculating the difference between Q tables obtained through the file /tmp/q_tables
    for data in input_data:
        data_object = Data()
        data_object.loadTables(data[0])
        print("     Length of Tables originally: {0}".format(data_object.qtables_length))
        data_object.pruneTables(20)
        print("     Length of tables after pruning: {0}".format(data_object.qtables_length))
        data_object.analyseN()
        data_object.analyseQ()
        if show_results:
            data_object.plotData(data[2], XLIMIT)
        data_object.saveasCSV(data[0], data[1])

    plt.show()
