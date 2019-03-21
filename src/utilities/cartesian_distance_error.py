#! /usr/bin/python
from line_generator import LineGenerator
import distance_from_line as dist_utils

class CartesianDistanceError(object):
    def __init__(self):
        self.line_generator = LineGenerator()

    def load_line(self, path):
        pipe_points, pipe_thetas = self.line_generator.load_line_points(path)
        return pipe_points, pipe_thetas

    def calc_error_from_line(self, pos):
        closest_point_idx = self.line_generator.idx_of_closet_point(pos)
        if closest_point_idx == 0:
            closest_points_idxs = [closest_point_idx, closest_point_idx + 1]
        elif closest_point_idx == self.line_generator.num_segs - 1:
            closest_points_idxs = [closest_point_idx - 1, closest_point_idx]
        else:
            closest_points_idxs = [closest_point_idx - 1, closest_point_idx + 1]

        distance_from_line = dist_utils.perpendicular_distance(pos,
                        self.line_generator.points[closest_points_idxs[0]][:2],
                        self.line_generator.points[closest_points_idxs[1]][:2])
        return distance_from_line

    def calc_error_from_line_signed(self, pos):
        closest_point_idx = self.line_generator.idx_of_closet_point(pos)
        if closest_point_idx == 0:
            closest_points_idxs = [closest_point_idx, closest_point_idx + 1]
        elif closest_point_idx == self.line_generator.num_segs - 1:
            closest_points_idxs = [closest_point_idx - 1, closest_point_idx]
        else:
            closest_points_idxs = [closest_point_idx - 1, closest_point_idx + 1]

        distance_from_line = dist_utils.perpendicular_distance_signed(pos,
                        self.line_generator.points[closest_points_idxs[0]][:2],
                        self.line_generator.points[closest_points_idxs[1]][:2])
        return distance_from_line
