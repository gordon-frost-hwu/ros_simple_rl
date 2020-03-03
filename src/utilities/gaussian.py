#! /usr/bin/env python
""" This module provides a function which takes the sum of N gaussian distributions """
import roslib;roslib.load_manifest("ros_simple_rl")
import rospy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sys
from rospkg import RosPack
from math import log, sqrt
from load_goals import load_goals
rospack = RosPack()

SCALE_OUTPUT = False

# List of goals to calculate distances etc... [North, East, Depth, Class]
# "Class" of goal determines whether the point is of interest or should be feared etc...
GOAL_POSITIONS = load_goals()
POINT_TYPE_IDX = -1    # minus one due to indexing of last element

def gaussian_generatorMAP(x,y):
    cnt = 0
    while cnt < len(GOAL_POSITIONS):
        x0 = GOAL_POSITIONS[cnt][0]; y0 = GOAL_POSITIONS[cnt][1]
        variance_x = GOAL_POSITIONS[cnt][-3]; variance_y = GOAL_POSITIONS[cnt][-2]
        #print GOAL_POSITIONS[cnt]
        if GOAL_POSITIONS[cnt][POINT_TYPE_IDX] == 1:
            mu = 1; sigma = [variance_x, variance_y]    #[6 for i in range(2)]
        elif GOAL_POSITIONS[cnt][POINT_TYPE_IDX] == 2:
            mu = 0.4; sigma = [variance_x, variance_y]    #[2 for i in range(2)]
        elif GOAL_POSITIONS[cnt][POINT_TYPE_IDX] == 3:
            mu = -0.2; sigma = [variance_x, variance_y]    #[2 for i in range(2)]    # -ve mu due to being fear
        yield gaussian2D(x, y, x0, y0, mu, sigma)
        cnt += 1
#TODO: Fear value should only come from the closest fear point. i.e. determine values for each and return the max
def gaussian_generatorFEAR(x, y):
    cnt = 0
    # Get only the type 3 points from GOAL_POSITIONS
    fear_points = GOAL_POSITIONS[np.where(GOAL_POSITIONS[:,POINT_TYPE_IDX] == 3),:][0]
    # keep value between [0, 1]
    if len(fear_points.shape) != 1:
        scale_divisor = fear_points.shape[0]
    else:
        scale_divisor = 1.0
    while cnt < fear_points.shape[0]:
        x0 = fear_points[cnt, 0]; y0 = fear_points[cnt, 1]
        mu = 1.0; sigma = [fear_points[cnt, -3], fear_points[cnt, -2]]      #[2 for i in range(2)]
        yield (gaussian2D(x, y, x0, y0, mu, sigma) / scale_divisor)
        cnt += 1

def gaussian_generatorFEARMax(x, y):
    """ return only the highest fear value which will be elicited by the closest fearful point in points_of_interest """
    # Get only the type 3 points from GOAL_POSITIONS
    fear_points = GOAL_POSITIONS[np.where(GOAL_POSITIONS[:,POINT_TYPE_IDX] == 3),:][0]
    cnt = 0
    max_fear = 0.0
    while cnt < fear_points.shape[0]:
        x0 = fear_points[cnt, 0]; y0 = fear_points[cnt, 1]
        mu = 1.0; sigma = [fear_points[cnt, -3], fear_points[cnt, -2]]      #[2 for i in range(2)]
        fear_intensity_from_point = gaussian2D(x, y, x0, y0, mu, sigma)
        if  fear_intensity_from_point > max_fear:
            max_fear = fear_intensity_from_point
        cnt += 1
    return max_fear

def gaussian_generatorPERCEPTION(x, y):
    # This function used if the gaussian is to be applied to some distance readings from a sonar.
    # ie. some points have been classified (somehow!) as fearful or points of interest. Therefore, they should
    # be used to find the worth of your current location (based on perception readings, not location in a MAP).
    # :arguments: x = list of distances to points of interest; y = list of distances to points we should fear
    cnt = 0
    #all_points = [x, y]
    #while cnt < len(x):
    yield (gaussian1D(0, x, 1, 8) + gaussian1D(0, y, -0.4, 2))


def gaussian2D(x, y, x0, y0, mu, sigma):
    """
    :param x: north position
    :param y: east position
    :param x0: centre of Normal distribution in North
    :param y0: centre of Normal distribution in East
    :param mu: Height of normal distribution
    :param sigma: 2 dimensional list of form [x_sigma, y_sigma]
    :return: value of point x,y wrt normal distribution centred on x0,y0
    """
    assert len(sigma) == 2
    x_term = (x - x0)**2 / (2 * float(sigma[0])**2)
    y_term = (y - y0)**2 / (2 * float(sigma[1])**2)
    res = mu * np.exp( -(x_term + y_term))
    return res

def gaussian1D(x, x0, mu, sigma):
    # make sure everything is of type float so that calculation is correct
    x = float(x); x0 = float(x0); mu = float(mu); sigma = float(sigma)
    x_term = (x - x0)**2 / (2 * sigma**2)
    return (mu * np.exp(-x_term))

def inverseGaussian(probability, x0, mu, sigma):
    x0 = float(x0); mu = float(mu); sigma = float(sigma)
    return sqrt(-(2 * sigma**2) * (log(mu * probability)) + x0)

if __name__== '__main__':
    arg = sys.argv[1]
    print("Plotting %s Function" % arg)
    PLOTTING_GRID_RESOLUTION = 0.01
    if arg == "happy":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #plt.ion()
        X = np.arange(0, 20, PLOTTING_GRID_RESOLUTION)
        Y = np.arange(0, 20, PLOTTING_GRID_RESOLUTION)
        X, Y = np.meshgrid(X, Y)
        print("X in input (1st row only):\n%s" % X[0])
        print("----")
        print("Y in input (1st row only):\n%s" % Y[0])
        # R = np.sqrt(X**2 + Y**2)
        # Z = np.sin(R)
        Z = np.zeros([X.shape[0], Y.shape[0]])
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                Z[i][j] = sum(gaussian_generatorMAP(X[0][i], Y[j][1]))

        max_happiness_level = np.max(Z)
        print("Max Happiness Level is: %s" % max_happiness_level)
        north_location = np.where(Z == max_happiness_level)[0][0] * PLOTTING_GRID_RESOLUTION
        east_location = np.where(Z == max_happiness_level)[1][0] * PLOTTING_GRID_RESOLUTION
        print("Occurs at grid location: [%s, %s]" % (north_location, east_location))

        # If the raw output is above zero, scale the whole output space
        # This will not work if just querying one point at a time though!!
        if SCALE_OUTPUT:
            if np.max(Z) > 1:
                Z = Z / np.max(Z)
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
            linewidth=0, antialiased=False)
        plt.show()

    elif arg == "fear":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #plt.ion()
        X = np.arange(-1, 1, PLOTTING_GRID_RESOLUTION)
        Y = np.arange(-3.14, 3.14, PLOTTING_GRID_RESOLUTION)
        X, Y = np.meshgrid(X, Y)
        print("X in input (1st row only):\n%s" % X[0])
        print("----")
        print("Y in input (1st row only):\n%s" % Y[0])
        # R = np.sqrt(X**2 + Y**2)
        # Z = np.sin(R)
        Z = np.zeros([X.shape[0], Y.shape[0]])
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                # Z[i][j] = sum(gaussian_generatorFEARMax(X[0][i], Y[j][1]))
                Z[i][j] = X[0][i] / Y[j][1]
                #Z[i][j] = (gaussian_generatorFEARMax(X[0][i], Y[j][1]))

        # If the raw output is above zero, scale the whole output space
        # This will not work if just querying one point at a time though!!
        if SCALE_OUTPUT:
            if np.max(Z) > 1:
                Z = Z / np.max(Z)
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
            linewidth=0, antialiased=False)
        plt.show()
