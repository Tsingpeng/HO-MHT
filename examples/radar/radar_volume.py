import numpy as np

from mht.scan_volume import (Volume)

class Sector(Volume):

    def __init__(self, angle_start, angle_end, range_max, P_D, clutter_lambda, init_lambda):
        super(Sector, self).__init__(P_D, clutter_lambda, init_lambda)
        assert(-np.pi < angle_start <= np.pi)
        assert(-np.pi < angle_end <= np.pi)
        assert(range_max > 0.0)
        self._angle_start = angle_start
        self._angle_end = angle_end
        self._range = range_max

    def _d_angle(self):
        if self._angle_start >= self._angle_end:
            return self._angle_start - self._angle_end
        else:
            return 2.0*np.pi - (self._angle_end - self._angle_start)

    def volume(self):#curve's length
        return self._d_angle() * self._range

    def _is_bearing_within(self, b):
        if self._angle_start >= self._angle_end:
            return self._angle_end <= b <= self._angle_start
        else:
            return (b <= self._angle_start) or (b >= self._angle_end)

    def is_within(self, z):
        ''' z[0] is range, z[1] is bearing in (-pi,pi] '''
        return self._is_bearing_within(z[1]) and (0.0 <= z[0] <= self._range)

    def scan(self, objects, measmodel):
        r = [0.0, self._range]
        if self._angle_start >= self._angle_end:
            range_c = np.array([
                r,
                [self._angle_end, self._angle_start]
            ])
            return super(Sector, self).scan(objects, measmodel, range_c)
        else:
            lower_range_c = np.array([
                r,
                [-np.pi, self._angle_start]
            ])
            upper_range_c = np.array([
                r,
                [self._angle_end, np.pi]
            ])
            union = super(Sector, self).scan(objects, measmodel, lower_range_c)
            union.extend(super(Sector, self).scan(objects, measmodel, upper_range_c))
            return union