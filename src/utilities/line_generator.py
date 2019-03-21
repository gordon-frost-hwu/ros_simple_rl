#! /usr/bin/python
from scipy import random
from copy import deepcopy
import numpy as np

sin = np.sin
cos = np.cos

def noisy_variable(var, sig):
    return var + np.random.normal(0.0, sig)

class LineGenerator(object):
    points = []
    thetas = []
    num_segs = 5
    _idx_of_closet_point = 0

    def generate_line(self, seg_length=1.0, start_point=[0.0, 0.0], first_seg_heading=0.0, num_segs=15):
        """return: points of line in numpy array """
        # seg_length = 5.0
        # start_point = [0.0, 0.0]    # [north, east]
        # first_seg_heading = 0.0
        # num_segs = 15
        self.num_segs = num_segs
        self.points.append(start_point)
        for i in range(num_segs):
            if i == 0:
                theta = first_seg_heading
            # calculate the next north-east points based on the latest point and desired heading etc.
            # for when heading is in quadrant 1: [0, 1.57]
            if theta == 0.0:
                north_next = self.points[-1][0] + seg_length
                east_next = self.points[-1][1]
            elif theta == 1.57:
                north_next = self.points[-1][0]
                east_next = self.points[-1][1] + seg_length
            elif theta == -1.57:
                north_next = self.points[-1][0]
                east_next = self.points[-1][1] - seg_length
            elif theta == -np.pi or theta == np.pi:
                north_next = self.points[-1][0] - seg_length
                east_next = self.points[-1][1]
            elif theta > 0.0 and theta < 1.57:
                north_next = self.points[-1][0] + seg_length * cos(theta)
                east_next = self.points[-1][1] + seg_length * sin(theta)
            elif theta < 0.0 and theta > -1.57:
                phi = 1.57 - abs(theta)
                north_next = self.points[-1][0] + seg_length * sin(phi)
                east_next = self.points[-1][1] - seg_length * cos(phi)
            elif theta < -1.57:
                phi = 1.57 - (abs(theta) - 1.57)
                north_next = self.points[-1][0] - seg_length * cos(phi)
                east_next = self.points[-1][1] - seg_length * sin(phi)
            elif theta > 1.57:
                phi = 3.14 - theta
                # print("theta > 1.57: costheta: {0}".format(cos(phi)))
                north_next = self.points[-1][0] - (seg_length * cos(phi))
                east_next = self.points[-1][1] + seg_length * sin(phi)

            # add the newly calculated point to the list of points
            self.points.append([north_next, east_next])

            self.thetas.append(theta)

            # choose the heading of the next line segment
            theta = noisy_variable(theta, 0.2)
            seg_length = noisy_variable(seg_length, 0.6)

            if theta > np.pi:
                theta = - ((2 * np.pi) - theta)
            elif theta < -np.pi:
                theta = (2 * np.pi) - theta
            print(theta)

        # add a final theta so that thetas and points are the same length
        self.thetas.append(0.0)

        return np.array(self.points), self.thetas

    def load_line_points(self, path):
        print("Loading line described by points in file: {0}".format(path))
        data = np.loadtxt(path, dtype=float, delimiter=" ", comments="#")
        self.points = data[:,:6]
        self.thetas = data[:, 6]
        self.num_segs = data.shape[0]
        print(self.points)
        print(self.thetas)
        print("Load completed ...")
        return self.points, self.thetas

    def save_points(self, file_path):
        with open(file_path, "w") as f:
            idx = 0
            f.write("#north, east, depth, roll, pitch, yaw, angle-to-next-point\n")
            for point in self.points:
                f.write("{0} {1} 0 0 0 0 {2}\n".format(point[0], point[1], self.thetas[idx]))
                idx += 1

    def idx_of_closet_point(self, point):
        self.nearest_point_on_line(point)
        return self._idx_of_closet_point

    def nearest_point_on_line(self, vehicle_pos):
        points = np.array(self.points)[:,:2]
        # print("vehicle Pos: {0}".format(vehicle_pos))
        # print("POoints: {0}".format(self.points))
        tmp = (vehicle_pos - points)**2
        tmp1 = deepcopy(tmp.sum(axis=1))
        # print("nearest_point_on_line:")
        # print(tmp1)
        # print("{0}".format(np.where(tmp1==tmp1.min(axis=0))[0][0]))
        self._idx_of_closet_point = np.where(tmp1==tmp1.min(axis=0))[0][0]
        return points[self._idx_of_closet_point], self.thetas[self._idx_of_closet_point]
