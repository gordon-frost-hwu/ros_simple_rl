#! /usr/bin/python
import numpy as np
import sys

# Log file must have form:
#     | #col1,col2,col3,colN
#     | 0,0,0,0
#     | 1,4,43,4
#     | 6,8,2,44

def perpendicular_distance(p, P1, P2):
    return abs((P2[0] - P1[0])*p[1] - (P2[1] - P1[1]) * p[0] + (P2[1]*P1[0]) - (P2[0]*P1[1])) / \
           (np.sqrt((P2[0] - P1[0])**2 + (P2[1] - P1[1])**2))

def perpendicular_distance_signed(p, P1, P2):
    return ((P2[0] - P1[0])*p[1] - (P2[1] - P1[1]) * p[0] + (P2[1]*P1[0]) - (P2[0]*P1[1])) / \
           (np.sqrt((P2[0] - P1[0])**2 + (P2[1] - P1[1])**2))

def perp_distance_list(points, P):
    pass

if __name__ == '__main__':

    P1 = [0, 0]
    P2 = [4, 4]
    p = [3.3, 3]
    points = np.array([[1, 3], [2.0, 2.0], [2.4, 2.2]])

    print(perpendicular_distance(p, P1, P2))