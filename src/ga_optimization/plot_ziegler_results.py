#!/usr/bin/python

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import numpy as np
import sys
import argparse

FILES = {
    "Marginally stable": 'ziegler/validate_ziegler_pid4/actions0.csv',
    "P": 'ziegler/validate_ziegler_pid4/actions1.csv',
    "PI": 'ziegler/validate_ziegler_pid4/actions2.csv',
    "PID": 'ziegler/validate_ziegler_pid4/actions3.csv',
}

FILES2 = {
    "Marginally stable": 'ziegler/validate_ziegler_pid7/actions0.csv',
    "P": 'ziegler/validate_ziegler_pid7/actions1.csv',
    "PI": 'ziegler/validate_ziegler_pid7/actions2.csv',
    "PID": 'ziegler/validate_ziegler_pid7/actions3.csv',
}


class PlotStuff(object):
    def __init__(self, args):
        self._name_history = {}
        self.args = args
        self.colors = ['r', 'g', 'b', 'k']
        self._plot_idx = 1

        self._plotted_objects = []
        self._plotted_objects_2 = []

    def run(self):
        # Divide the figure into a 3x1 grid, and give me the first section
        fig = plt.figure()
        # gs = GridSpec(1, 1, figure=fig)
        # self.ax1 = fig.add_subplot(gs[0, :])
        # self.axes = [self.ax1]

        gs = GridSpec(2, 2, figure=fig)
        self.ax1 = fig.add_subplot(gs[0, 0])
        self.ax2 = fig.add_subplot(gs[0, 1])
        self.ax3 = fig.add_subplot(gs[1, 0])
        self.ax4 = fig.add_subplot(gs[1, 1])
        self.axes = [self.ax1, self.ax2, self.ax3, self.ax4]

        for key in sorted(FILES.keys()):
            df = pd.read_csv(FILES[key], self.args.delimiter)
            ax = self.ax1 if key == "Marginally stable" else self.ax2
            self.plot(df, key, ax)
        
        for key in sorted(FILES2.keys()):
            df = pd.read_csv(FILES2[key], self.args.delimiter)
            ax = self.ax3 if key == "Marginally stable" else self.ax4
            self.plot(df, key, ax, add_to_legend=False)

        # Set common labels
        fig.text(0.5, 0.04, 'Step Number', ha='center', va='center', fontsize=35)
        fig.text(0.06, 0.5, '$\Theta$ (normalised)', ha='center', va='center', rotation='vertical', fontsize=35)

        legend = plt.legend(handles=self._plotted_objects, 
                            title='Legend',
                            fontsize=26)
        plt.setp(legend.get_title(),fontsize=35)

        for ax in self.axes:
            ax.grid(which='major', linestyle='-', linewidth='1.0', color='black')

        for ax in self.axes:
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(20) 
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(20)
        
        self.ax1.set_title(r'Response with $K_{crit} = 0.8$', fontsize=24, fontweight='bold')
        self.ax2.set_title(r'Responses for P, PI, and PID controllers, $K_{crit} = 0.8$', fontsize=24, fontweight='bold')
        self.ax3.set_title(r'Response with $K_{crit} = 1.1$', fontsize=24, fontweight='bold')
        self.ax4.set_title(r'Responses for P, PI, and PID controllers, $K_{crit} = 1.1$', fontsize=24, fontweight='bold')
    
    def plot(self, df, key, ax, add_to_legend=True):
        x = df.index
        # df.plot(x=x, y="Mean", ax=ax, linewidth=4, yerr="Std_Dev")
        color_idx = self._plot_idx % len(self.colors)
        print("color_idx: {0} = {1}".format(self._plot_idx, color_idx))
        plotted_obj, = ax.plot(x, df["#angle"], color=self.colors[color_idx], linestyle="-", linewidth=4, label="{0}".format(key))
        
        if add_to_legend:
            self._plotted_objects.append(plotted_obj)
        
        # ax.plot(x, df["Lower_Std"], color=self.colors[self._plot_idx], linestyle="--", linewidth=2)
        # ax.plot(x, df["Upper_Std"], color=self.colors[self._plot_idx], linestyle="--", linewidth=2)

        self._plot_idx += 1
        
    
    # def axes_iterable(self):
    #     return self.axes

    # def setup_for_print(self):
    #     # Customize appearance of the subplots/figure
    #     if self.args.xlim is not None:
    #         for ax in self.axes_iterable():
    #             ax.set_xlim(self.args.xlim[0], self.args.xlim[1])
        
    #     for ax in self.axes_iterable():
    #         ax.grid(which='major', linestyle='-', linewidth='1.0', color='black')

    #     for ax in self.axes_iterable():
    #         for tick in ax.xaxis.get_major_ticks():
    #             tick.label.set_fontsize(20) 
    #         for tick in ax.yaxis.get_major_ticks():
    #             tick.label.set_fontsize(20)
        
    #     print("original labels: {0}".format(self._name_history))
    #     for ax in self.axes_iterable():
    #         ax.legend([self.convert_label(label) for label in self._name_history[ax]],
    #                 fontsize=20,
    #                 # loc="lower right",f
    #                 ncol=3)#len(self._name_history[ax]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to plot specific reward variations")

    parser.add_argument(
        "--delimiter", default='\t', help="Whether to execute the experimental custom plotting code",
    )
    parser.add_argument("--xlim", type=int, nargs=2, metavar=["low", "high"], help="X axis start and end")

    # parser.add_argument("files", nargs='+', help="Name of the ROS bag to load")

    args = parser.parse_args()
    print(args)

    p = PlotStuff(args)
    p.run()
    
    plt.show()