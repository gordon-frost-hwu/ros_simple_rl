#! /usr/bin/env python
''' Utils contains (mostly geometric) functions for general use)'''

import numpy as np

from math import pi
import math
import copy
#import pynotify

def getIndices(items, full_list):
    indices = []
    for item in items:
        indices.append(full_list.index(item))
    return indices

def send_ubuntuMessage(title, message):
    pynotify.init("Utils_Notify")
    notice = pynotify.Notification(title, message)
    notice.show()
    return

def epsilonEquals(a,b,e):
    ''' boolean: is "a" equal to "b" within a tolerance of +- "e"'''
    return a < b + e and a > b - e    

def epsilonEqualsMod2Pi(a,b,e):
    ''' boolean: is "a" equal to "b" within a tolerance of +- "e" where a and b represent radian angles'''

    #drive into radian polar range
    while a > pi: a -= math.pi*2
    while a <-pi: a += math.pi*2
    while b > pi: b -= math.pi*2
    while b <-pi: b += math.pi*2

    #measure difference
    error = abs(a-b)
    #if absolute error is above pi, it means we are measuring it in the wrong direction
    if error > pi : error = 2*pi - error

    return error < e

def epsilonEqualsNED(a,b,e, depth_e=None):
    ''' boolean: is "a" equal to "b" within a tolerance of +- "e", where A and B are north, east, depth datatypes (auv_msgs/NED)'''
    if depth_e == None:
        depth_e = e
        
    return  epsilonEquals(a.north, b.north, e) and \
            epsilonEquals(a.east,  b.east,  e) and \
            epsilonEquals(a.depth, b.depth, depth_e)

def epsilonEqualsRPY(a,b,e):
    ''' boolean: is "a" equal to "b" within a tolerance of +- "e", where A and B are roll, pitch, yaw datatypes (auv_msgs/RPY)'''

    return  epsilonEqualsMod2Pi(a.roll,  b.roll, e) and\
            epsilonEqualsMod2Pi(a.pitch, b.pitch,e) and\
            epsilonEqualsMod2Pi(a.yaw,   b.yaw,  e)
            
def epsilonEqualsPY(a,b,e):
    ''' boolean: is "a" equal to "b" within a tolerance of +- "e", where A and B are pitch, yaw but NOT yaw (auv_msgs/RPY)'''
    return  epsilonEqualsMod2Pi(a.pitch, b.pitch,e) and\
            epsilonEqualsMod2Pi(a.yaw,   b.yaw,  e)

def epsilonEqualsY(a,b,e):
    ''' boolean: is "a" equal to "b" within a tolerance of +- "e", where A and B are pitch, yaw but NOT yaw (auv_msgs/RPY)'''
    return  epsilonEqualsMod2Pi(a.yaw,   b.yaw,  e)

            
def dist(a,b):
    '''Eucledian distance between 2 n-dimentional vectors expressed as tuples'''
    from math import sqrt
    sum = 0
    for (a_i, b_i) in zip(a,b):
        sum+= (a_i - b_i) ** 2
    return sqrt(sum)

def distance_betweenObjects(list_of_wps, current_wp):
    """ input is a list of VehiclePose objects and object of the current wp as vehicle pose object """
    measure = 10000     # just a randomly high number, higher than any calculated number to begin with.
    pos_wps = []
    curr_wp = (current_wp.position.north, current_wp.position.east, current_wp.position.depth)

    # TODO: fix the fact that the first wp will always be considered an option, even if it is the further away!
    # step through the list, calculating the distance of the point to the current_wp, if it is less, save that as a
    # potential solution point.
    for pose in list_of_wps:
        possible_wp = (pose.position.north, pose.position.east, pose.position.depth)
        d = dist(curr_wp, possible_wp)
        #print "Comparing distance, %s, with measure, %s" % (d, measure)
        if d < measure or d == measure:
            pos_wps.append(pose)
            measure = d

    # out of the list of closest points, all should have the same NED, choose one at random
    #goal_wp = pos_wps[random.randint(0, len(pos_wps)-1)]
    goal_wp = pos_wps[-1]
    #print "List of waypoints which satisfy the criterion: ", pos_wps
    return goal_wp

def find_words(text, search):
    ''' Searches for exact match of word in a string, returns false if no match, number of matching words if they are in string'''
    Texts = text.split()
    Searches = search.split()
    lenText = len(text)
    lensearch = len(search)

    for Text in Texts:
        for Search in Searches:
            if hash(Text) == hash(Search):
                return True
    return False

if __name__ == '__main__':
    print "TESTING utils.py"
    D = ((0,0,0),
         (1,2,3),
         (2,2,2),
         (5,5,5),
         (10,20,30),
         (-1,-1,-1),
         )

    #x = np.array([[0],[1],[2]])
    x = (0,1,2)


    print knn_search(x, D, 3)
    print knn_search(x, D, 1)
