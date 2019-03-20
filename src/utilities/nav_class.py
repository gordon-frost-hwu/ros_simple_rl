import roslib; roslib.load_manifest("ros_simple_rl")
from auv_msgs.msg import NavSts
from math import radians, cos, sin, asin, sqrt
import rospy

class Nav(object):
    # Create the msg and subscriber as Class attributes. If they are in __init__ method and Nav class is inherited, they
    # will otherwise not be created/instanciated
    _nav = NavSts()

    def __init__(self):
        self.nav_sub = rospy.Subscriber("/nav/nav_sts", NavSts, self._navCallback)

    def _navCallback(self, msg):
        self._nav = msg
    
    def distance_to_target(self, target_location_ned):
        north = self._nav.global_position.latitude
        north = self._nav.global_position.longnitude
    
    def distance_between_two_global_positions(self, lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians 
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        # haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        km = 6367 * c
        
        R = 6371  # radius of the earth in km
        x = (lon2 - lon1) * cos( 0.5*(lat2+lat1) )
        y = lat2 - lat1
        d = R * sqrt( x*x + y*y )
        # Return the value in metres
        #return (float(km) / 1000.0)
        return km, d

if __name__=='__main__':
    lat1 = 55.911442
    lon1 = -3.326416
    lat2 = 55.910792
    lon2 = -3.327918
    
    nav = Nav()
    km, d = nav.distance_between_two_global_positions(lon1,lat1,lon2,lat2)
    print("Distance: {0} {1}".format(km, d))
