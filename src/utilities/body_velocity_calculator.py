#! /usr/bin/python

import roslib; roslib.load_manifest("ros_simple_rl")
import rospy
from auv_msgs.msg import NED, RPY, VehiclePose
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import quaternion_from_euler
from numpy import pi, deg2rad, rad2deg, sqrt, arctan
import sys
from nav_class import Nav
NINETY_DEGREES = pi / 2.0
WORLD_FRAME = "/map"

def calculate_body_velocity(surge_velocity, sway_velocity, vehicle_heading):
    # guard against division by zero errors
    if surge_velocity == 0.0 or sway_velocity == 0.0:
        if surge_velocity == 0.0:
            theta = 90.0
        elif sway_velocity == 0.0:
            theta = 0.0
    else:
        # calculate the angles between X, Y, and the resultant velocity vector, R
        theta = arctan(abs(surge_velocity) / abs(sway_velocity))
    psi = deg2rad(90.0) - theta

    #print("Theta in degrees: {0}".format(rad2deg(theta)))
    #print("Psi in degrees: {0}".format(rad2deg(psi)))
    resultant_velocity = sqrt(surge_velocity**2 + sway_velocity**2)
    #print("Resultant Velocity: {0}".format(resultant_velocity))

    if surge_velocity >= 0.0 and sway_velocity >= 0.0:
        # TODO: Checked - is ok!    checked 14/04/15
        J = vehicle_heading + psi
    elif surge_velocity >= 0.0 and sway_velocity < 0.0:
        # TODO: Checked - is ok!    checked 14/04/15
        if vehicle_heading >= 0.0:
            J = vehicle_heading - psi
        elif vehicle_heading < 0.0:
            J = vehicle_heading - psi
    elif surge_velocity < 0.0 and sway_velocity >= 0.0:
        #TODO: Checked - is ok!     checked 14/04/15
        if vehicle_heading >= 0.0:
            J = vehicle_heading + theta + NINETY_DEGREES
        elif -NINETY_DEGREES < vehicle_heading < 0.0:
            J = pi - abs(vehicle_heading) - psi
        elif vehicle_heading < -NINETY_DEGREES:
            J = vehicle_heading + NINETY_DEGREES + theta
    elif surge_velocity < 0.0 and sway_velocity < 0.0:
        # TODO: Checked - is ok!    checked 14/04/15
        if vehicle_heading > 0.0:
            # print("hello")
            # print("vehicle heading: {0}".format(vehicle_heading))
            # print("psi: {0}".format(psi))
            # print("theta: {0}".format(theta))
            # print("90: {0}".format(NINETY_DEGREES))
            J = -((NINETY_DEGREES - vehicle_heading) + theta)
        elif -NINETY_DEGREES < vehicle_heading <= 0.0:
            J = vehicle_heading - NINETY_DEGREES - theta
        elif vehicle_heading < -NINETY_DEGREES:
            J = vehicle_heading + pi + psi

    resultant_velocity_heading = J
    #print("Resultant Velocity Heading in degrees: {0}".format(rad2deg(resultant_velocity_heading)))

    try:
        assert (0.0 <= abs(resultant_velocity_heading) <= pi)
    except AssertionError:
        #print("Clamping/Wrapping Heading output: {0}".format(J))
        if resultant_velocity_heading > pi:
            resultant_velocity_heading = -pi + (resultant_velocity_heading - pi)
        elif resultant_velocity_heading < -pi:
            resultant_velocity_heading = (2 * pi) + resultant_velocity_heading

    #print("Clamped Velocity Heading in degrees: {0}".format(rad2deg(J)))

    return resultant_velocity, resultant_velocity_heading

# Class used only if module is being run as a ros node
class BodyVelocityCalculator(Nav):
    array_id = 0

    def __init__(self):
        Nav.__init__(self)
        self.wp_rviz_publisher = rospy.Publisher("/vehicle_body_velocity_markers", MarkerArray)

    def loop(self):
        if self._nav is not None:
            velocity, heading = calculate_body_velocity(self._nav.body_velocity.x, self._nav.body_velocity.y,
                                                        self._nav.orientation.yaw)
            print("Body Velocity:    {0}".format(velocity))
            print("Velocity Heading: {0}".format(heading))
            pose = VehiclePose(
                    position=NED(north=self._nav.position.north,east=self._nav.position.east,
                                 depth=self._nav.position.depth),
                    orientation=RPY(roll=0,pitch=0,yaw=heading))

            # publish an arrow marker to according to above calculation
            self.rviz_points_array(pose)

    def rviz_points_array(self, point, draw_arrows=True):
        #print "RVIZ_POINTS_ARRAY: Contents of point: ", point
        # Create the Spheres representing the waypoints
        self.wp_spheremarkerArray = MarkerArray()
        self.wp_spheremarker = Marker()
        self.wp_spheremarker.header.frame_id = WORLD_FRAME
        self.wp_spheremarker.type = self.wp_spheremarker.SPHERE
        self.wp_spheremarker.action = self.wp_spheremarker.ADD
        if (draw_arrows == True):
            self.wp_spheremarker.scale.x = 0.4
            self.wp_spheremarker.scale.y = 0.4
            self.wp_spheremarker.scale.z = 0.4
            self.wp_spheremarker.color.a = 1.0
            self.wp_spheremarker.color.r = 1.0
            self.wp_spheremarker.color.g = 0.0
            self.wp_spheremarker.color.b = 0.0
        elif (draw_arrows == False):
            self.wp_spheremarker.scale.x = 0.4
            self.wp_spheremarker.scale.y = 0.4
            self.wp_spheremarker.scale.z = 0.4
            self.wp_spheremarker.color.a = 1.0
            self.wp_spheremarker.color.r = 0.0
            self.wp_spheremarker.color.g = 0.0
            self.wp_spheremarker.color.b = 0.0

        self.wp_spheremarker.pose.orientation.w = 1

        self.wp_spheremarker.id = self.array_id
        self.wp_spheremarker.pose.position.x = point.position.north
        self.wp_spheremarker.pose.position.y = -point.position.east
        self.wp_spheremarker.pose.position.z = -point.position.depth
        self.wp_spheremarkerArray.markers.append(self.wp_spheremarker)
        self.array_id = self.array_id + 1
        # Publish the Sphere Marker Array
        self.wp_rviz_publisher.publish(self.wp_spheremarkerArray)
        if (draw_arrows == True):
            # Create the Arrows for the waypoint orientations
            self.wp_arrowmarkerArray = MarkerArray()
            self.wp_arrowmarker = Marker()
            self.wp_arrowmarker.header.frame_id = WORLD_FRAME
            self.wp_arrowmarker.type = self.wp_arrowmarker.ARROW
            self.wp_arrowmarker.action = self.wp_arrowmarker.ADD
            self.wp_arrowmarker.scale.x = 0.6
            self.wp_arrowmarker.scale.y = 0.2
            self.wp_arrowmarker.scale.z = 0.3
            self.wp_arrowmarker.color.a = 1.0
            self.wp_arrowmarker.color.r = 1.0
            self.wp_arrowmarker.color.g = 0.0
            self.wp_arrowmarker.color.b = 0.0
            #may be incorrect transform here
            q = quaternion_from_euler(point.orientation.roll, point.orientation.pitch,-point.orientation.yaw, 'rxyz')

            self.wp_arrowmarker.pose.orientation.x = q[0]
            self.wp_arrowmarker.pose.orientation.y = q[1]
            self.wp_arrowmarker.pose.orientation.z = q[2]
            self.wp_arrowmarker.pose.orientation.w = q[3]

            self.wp_arrowmarker.id = self.array_id
            self.wp_arrowmarker.pose.position.x = point.position.north
            self.wp_arrowmarker.pose.position.y = -point.position.east
            self.wp_arrowmarker.pose.position.z = -point.position.depth
            self.wp_arrowmarkerArray.markers.append(self.wp_arrowmarker)
            self.array_id = self.array_id + 1
            # Publish the Arrow Marker Array
            self.wp_rviz_publisher.publish(self.wp_arrowmarkerArray)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        surge, sway, heading = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])

        calculate_body_velocity(surge, sway, heading)
    else:
        # run as a ros node to publish the calculated resultant velocity vector to Rviz
        rospy.init_node("body_velocity_calculator")
        body_velocity = BodyVelocityCalculator()
        while not rospy.is_shutdown():
            body_velocity.loop()
            rospy.sleep(1.0)
