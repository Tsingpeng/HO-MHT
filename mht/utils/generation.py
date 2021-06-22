import numpy as np
"""产生真实值和量测值"""
def _ground_truth_fixed_step(t_length, x_birth, t_birth, t_death, motionmodel, dt):
    """
    returns a list of length t_length with dictionaries containing object states.
    t_length:int 仿真时间; x_birth:list[int] 开始状态; t_birth:list[int] 开始时间;
    t_death:list[int] 消失时间; motionmodel:运动学模型(状态方程)类; dt:采样时间
    """

    trajs = [dict() for _ in range(t_length)]#trajectories:trajs[t]={i:state}
    #保证t_birth/t_death 不越界
    t_births = np.minimum(np.maximum(t_birth, 0), t_length - 1)#最小最大函数确定起始\终止时间
    t_deaths = np.minimum(np.maximum(t_death, 0), t_length - 1)

    for i, state in enumerate(x_birth):
        trajs[t_births[i]][i] = state#
        for t in range(t_births[i] + 1, t_deaths[i] + 1):
            trajs[t][i] = np.random.multivariate_normal(motionmodel.f(trajs[t-1][i], dt), motionmodel.Q(dt))

    return trajs


def ground_truth(t_end, x_birth, t_birth, t_death, motionmodel, dt=1.0):
    """
    returns a list of length floor(t_end/dt) with dictionaries containing object states. 
    """
    t_length = int(t_end/dt)
    t_births = [int(t/dt) for t in t_birth]
    t_deaths = [int(t/dt) for t in t_death]
    return _ground_truth_fixed_step(t_length, x_birth, t_births, t_deaths, motionmodel, dt)

def measurements(ground_truth, measmodel, P_D, lambda_c, range_c):
    """
    range_c is [[z0_min, z0_max], [z1_min, z1_max], ...]
    """
    meas = list(list() for _ in range(len(ground_truth)))#有多少行就代表多长时间

    for t, objects in enumerate(ground_truth):
        meas[t] = [
            measmodel.measure(state)
            for state in objects.values() if np.random.uniform() <= P_D
        ]

        for _ in range(np.random.poisson(lambda_c)):#服从泊松分布的杂波
            meas[t].append(measmodel.sample(range_c))

    return meas

def random_ground_truth(t_end, init_state_density, init_lambda, P_survival, motionmodel, dt):
    """
    Generate a set of ground truth objects. 
    Object birth is modelled as a Poisson process with init_lambda as the expected number of
    births per time step.
    The objects initial state is sampled from init_state_density and its trajectory follows
    motionmodel as long as it is still alive.
    Object death is modelled using a constant probability of survival P_survival.

    returns a list of length floor(t_end/dt) with dictionaries containing object states. 
    """
    t_length = int(t_end/dt)

    x_birth = list()
    t_birth = list()
    t_death = list()

    for t in range(t_length):
        for _ in range(np.random.poisson(init_lambda)):
            x_birth.append(init_state_density.sample())
            t_birth.append(t)

            t_d = t + 1
            while (np.random.uniform() <= P_survival) and (t_d < t_length):
                t_d += 1

            t_death.append(t_d)

    assert(len(x_birth)==len(t_birth)==len(t_death))
    assert((np.array(t_birth) < np.array(t_death)).all())

    return _ground_truth_fixed_step(t_length, x_birth, t_birth, t_death, motionmodel, dt)
