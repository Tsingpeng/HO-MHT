import numpy as np

from numpy import (sin, cos)

DT_DEFAULT=1.0

class ConstantVelocity2D:

    def __init__(self, sigma):
        self.sigma = sigma

    def dimension(self):
        return 4
    
    def F(self, x, dt=DT_DEFAULT):
        return np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1]
        ])

    def Q(self, dt=DT_DEFAULT):
        return np.array([
            [dt**4/4,      0, dt**3/2,       0],
            [0,      dt**4/4,       0, dt**3/2],
            [dt**3/2,      0,   dt**2,       0],
            [0,      dt**3/2,       0,   dt**2]
        ]) * (self.sigma**2)

    def f(self, x, dt=DT_DEFAULT):
        return np.dot(self.F(x, dt), x)

class CoordinatedTurn2D:

    def __init__(self, sigma_vel, sigma_angle_vel):
        G = np.array([[0, 0], [0, 0], [1, 0], [0, 0], [0, 1]])
        S = np.diag([sigma_vel**2, sigma_angle_vel**2])
        self.__Q = G @ S @ G.T
        #np.dot(np.dot(G, S), G.T)
        #G.np.dot(S).np.dot(G.T)

    def dimension(self):
        return 5

    def F(self, x, dt=DT_DEFAULT):
        return np.array([
            [1, 0, dt*cos(x[3]), -dt*x[2]*sin(x[3]),  0],
            [0, 1, dt*sin(x[3]),  dt*x[2]*cos(x[3]),  0],
            [0, 0,            1,                  0,  0],
            [0, 0,            0,                  1, dt],
            [0, 0,            0,                  0,  1]
        ])

    def Q(self, dt=DT_DEFAULT):
        return self.__Q

    def f(self, x, dt=DT_DEFAULT):
        dx = dt * np.array([
            x[2]*cos(x[3]),
            x[2]*sin(x[3]),
            0,
            x[4],
            0
        ])
        return x + dx
