import os
import sys

example_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(example_path, os.path.pardir))

import numpy as np
from tqdm import tqdm
from mht.tracker import (Tracker)

# Application implementations
from radar_target import (TargetRangeBearing_CV2D)
from radar_volume import (Sector)

from mht.utils import generation
from mht.utils import gaussian
from mht.utils import plot

###########
### MAIN ##
###########
targetmodel = TargetRangeBearing_CV2D

tracker = Tracker(
    max_nof_hyps = 50,
    hyp_weight_threshold = np.log(0.05),#when weight is larger than threshold->prone
)

dt = 0.1
init_lambda = 0.2#new born param.

ground_truth = generation.ground_truth(
    t_end = 100,
    x_birth = [
        np.array([10, -5,  1,  1]),
        np.array([10,  5,  1, -1])
    ],
    t_birth = [10, 0],
    t_death = [80, 70],
    motionmodel = targetmodel.motion(),
    dt = dt
)

volume = Sector(
    angle_start = np.pi/3,
    angle_end = -np.pi/3,
    range_max = 200.0,
    P_D = 0.90,
    clutter_lambda = 1.0,
    init_lambda = init_lambda
)

measurements = [volume.scan(objs, targetmodel.measure()) for objs in ground_truth]

estimations = list()
for t, detections in enumerate(tqdm(measurements)):
    t_now = dt * t
    estimations.append(tracker.process(detections, volume, targetmodel, t_now))

# Plot results
projection = lambda x: x[0:2]

p = plot.Plotter(to_plot_coordinates=projection)
p.trajectory_2d(ground_truth, linestyle='-')

inv_h = lambda z: np.array([z[0]*np.cos(z[1]), z[0]*np.sin(z[1]), 0.0, 0.0])
p.measurements_2d(measurements, inv_h, marker='.', color='k')
p.trajectory_2d(estimations, linestyle='--')
#p.covariances_2d(estimations, edgecolor='k', linewidth=1)
#p.legend(["1","0","3"])
p.show()
