import numpy as np

from mht.utils.gaussian import Density
from mht.models.target import Target
from mht.models.measmodel import RangeBearing
from mht.models.motionmodel import ConstantVelocity2D

from scipy.stats.distributions import (chi2)

class TargetRangeBearing_CV2D(Target):
    
    _motion = ConstantVelocity2D(sigma=0.1)
    _measure = RangeBearing(sigma_range=0.1, sigma_bearing=2.0*np.pi/180.0, pos=[0, 0])

    def __init__(self, density, t_now):
        super(TargetRangeBearing_CV2D, self).__init__(density, t_now)
        P_G = 0.99
        self._gating_size2 = chi2.ppf(P_G, self._measure.dimension())

    @staticmethod
    def _inv_h(z):
        return np.array([z[0]*np.cos(z[1]), z[0]*np.sin(z[1]), 0.0, 0.0])

    @classmethod
    def _P0(cls):
        return np.diag([1.0, 1.0, 1.0, 1.0])

    @classmethod
    def from_one_detection(cls, detection, t_now):
        return cls(
            density=Density(x=cls._inv_h(detection), P=cls._P0()),
            t_now=t_now
        )

    @classmethod
    def motion(self):
        return self._motion

    @classmethod
    def measure(self):
        return self._measure

    def gating(self, detections):
        return self._density.gating(detections, self._measure, self._gating_size2)

    def predicted_likelihood(self, detection):
        return self._density.predicted_likelihood(detection, self._measure)

    def max_coast_time(self):
        return 1.0
