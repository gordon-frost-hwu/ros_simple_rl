#! /usr/bin/python
import numpy as np

class AngleBetweenVectors(object):
    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::

                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        # v1_u = self.unit_vector(v1)
        # v2_u = self.unit_vector(v2)
        dot_product = np.dot(v1, v2)
        mag_v1 = np.linalg.norm(v1)
        mag_v2 = np.linalg.norm(v2)
        print([dot_product, mag_v1, mag_v2])
        angle = np.arccos(dot_product / (mag_v1 * mag_v2))
        print(angle)
        # if np.isnan(angle):
        #     if (v1_u == v2_u).all():
        #         return 0.0
        #     else:
        #         return np.pi
        return angle