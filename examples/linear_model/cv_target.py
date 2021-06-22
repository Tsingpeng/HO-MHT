from mht.models.target import Target
from mht.models.motionmodel import ConstantVelocity2D
from mht.models.measmodel import ConstantVelocity

from mht.utils.gaussian import (Density)

from scipy.stats.distributions import (chi2)#卡方分布
import numpy as np

class TargetPosition_CV2D(Target):
    
    _motion = ConstantVelocity2D(sigma=0.01)
    _measure = ConstantVelocity(sigma=0.1)

    def __init__(self, density, t_now):
        super(TargetPosition_CV2D, self).__init__(density, t_now)
        P_G = 0.99#门的大小取卡方分布(自由度为_measure.dimension())的P_G对应的分位数
        self._gating_size2 = chi2.ppf(P_G, self._measure.dimension())

    @staticmethod
    def _inv_h(z):
        return np.array([z[0], z[1], 0.0, 0.0])

    @classmethod
    def _P0(cls):
        R = cls._measure.R()
        return np.diag([R[0,0], R[1,1], 1.0, 1.0])

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
        return 4.0
