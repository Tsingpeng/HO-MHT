#coding utf-8
import os
import sys
"""Add this file as the main func."""
example_path = os.path.dirname(os.path.dirname(__file__))
# print(example_path)
# print(os.path.pardir)
# print(os.path.join(example_path, os.path.pardir))
sys.path.insert(0, os.path.join(example_path, os.path.pardir))
""" Start Main """
""" Import Modules """
import numpy as np
import matplotlib.pyplot as plt
from mht.tracker import (Tracker)

# Application implementation
from cv_target import (TargetPosition_CV2D)

from mht.scan_volume import (CartesianVolume)

from mht.utils import generation
from mht.utils import gaussian
from mht.utils import plot
from mht.utils import OSPA
import time
from tqdm import tqdm
""" Init """
targetmodel = TargetPosition_CV2D
measmodel = targetmodel.measure()#Here we use the classMethod to get the measmodel

init_mean = np.array([0.0, 0.0, 1.0, 0.0])
init_cov = np.diag([0.0, 10.0, 0.2, 0.0])

init_mean2 = np.array([500.0, 200.0, 0.0, -1.2])
init_cov2 = np.diag([10.0, 15.0, 0.50, 0.25])

init_mean3 = np.array([0.0, 200, 1, -1])
init_cov3 = np.diag([10.0, 10.0, 0.5, 0.25])
#init_lambda is used to get the object birth s.t. Possion(init_lambda) in <generation.random_ground_truth>
init_lambda = 0.5
#量测边界为ranges  检测概率P_D=0.9  生成杂波的泊松分布的系数为clutter_lambda=1.0
volume = CartesianVolume(
    ranges = np.array([[-1200.0, 1200.0], [-1200.0, 1200.0]]),
    P_D=0.95,
    clutter_lambda = 10,
    init_lambda = init_lambda
)
calc_time = list()#Record the total tracking time

show_plots = True

dt = 1.0#sampling time
nof_rounds = 1#Monte Carlo

Init_Stat = np.random.multivariate_normal(init_mean, init_cov)
Init_Stat2 = np.random.multivariate_normal(init_mean2, init_cov2)
Init_Stat3 = np.random.multivariate_normal(init_mean3, init_cov3)
""" M.C. """
for i in range(nof_rounds):
    """Get tracker,ground_truth and measurements"""
    end_time_eachMC = 1000
    tracker = Tracker(
        max_nof_hyps = 10,
        hyp_weight_threshold = np.log(0.05),
    )#跟踪器:_M=10,即m-best为10; 假设的权重阈值为ln(0.05)
    #P_survival: the const. prob. is used to model the object death time,i.e survival time.
    # ground_truth = generation.random_ground_truth(
    #     t_end = 100,
    #     init_state_density=gaussian.Density(x=init_mean, P=init_cov),
    #     init_lambda = init_lambda,
    #     P_survival = 0.95,
    #     motionmodel = targetmodel.motion(),
    #     dt = dt
    # )#真实值trajs[t]={i:state}:每次M.C.的running_time为t_end,初始的Density由init_mean/init_cov决定
    ##ground_truth(t_end, x_birth, t_birth, t_death, motionmodel, dt=1.0)
    ground_truth = generation.ground_truth(
        t_end= end_time_eachMC,
        x_birth= [Init_Stat, Init_Stat2, Init_Stat3],
        t_birth= [10,1,150],
        t_death= [470,500,1000],
        motionmodel= targetmodel.motion(),
        dt= dt
    )
    #Obtain mesurements based on ground truth
    measurements = [volume.scan(objs, measmodel) for objs in ground_truth]
    #print(measurements)
    #record the history of Estim.=>list({trid: Density})
    estimations = list()
    """Start Tracking"""
    tic = time.time()
    for t, detections in enumerate(tqdm(measurements)):
        t_now = dt * t#get the cur. time
        estimations.append(tracker.process(detections, volume, targetmodel, t_now))#Track and record
        #tracker.debug_print(t)
    """End Tracking"""
    print(estimations)
    calc_time.append(time.time()-tic)#记录跟踪的时间
    # get the track state=>{trid:xk}
    track_states = [
        {trid: density.x for trid, density in est.items()}
        for est in estimations
    ]
    print(ground_truth[6])
    print(track_states[6])
    """OPSA metric"""
    assert(len(track_states)==len(ground_truth))
    runtimes = len(track_states)
    ## OSPA_metric = OSPA.OSPA()
    rec_OPSA = list()
    for i in range(runtimes):
        score = OSPA.OSPA(ground_truth[i],track_states[i],1,2).solver()
        rec_OPSA.append(score)

    if show_plots:
        #show_plot = False
        q = plot.Plotter(to_plot_coordinates=measmodel.h, num=2)
        #q.trajectory_2d(ground_truth, linestyle='-')
        q.measurements_2d(measurements, marker='.', color='k')
        #q.trajectory_2d(estimations, linestyle='--')
        #q.covariances_2d(estimations, measmodel, edgecolor='k', linewidth=1)

        p = plot.Plotter(to_plot_coordinates=measmodel.h, num=1)
        p.trajectory_2d(ground_truth, linestyle='-')
        p.measurements_2d(measurements, marker='.', color='k')
        p.trajectory_2d(estimations, linestyle='--')
        p.covariances_2d(estimations, measmodel, edgecolor='k', linewidth=1)
        #p.legend()
        # paint OSPA
        p_o = plt.figure(num=3)
        p_o.gca().plot(np.arange(0,end_time_eachMC,dt),rec_OPSA)
        print(rec_OPSA)
        q.show()
        p.show()
        p_o.show()

print(time.time()-tic)
