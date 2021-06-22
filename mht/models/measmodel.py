import numpy as np

class ConstantVelocity:

    def __init__(self, sigma=1.0):
        #self.sigma = sigma
        self.__R = np.diag(2*[sigma**2])

    def dimension(self):
        return 2

    def H(self, x):
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]])

    def R(self):
        return self.__R

    def h(self, x):
        return np.dot(self.H(x), x)

    def measure(self, x):
        return np.random.multivariate_normal(self.h(x), self.__R)

    def sample(self, ranges):
        assert(len(ranges)==self.dimension())
        delta = ranges[:,1] - ranges[:,0]
        return ranges[:,0] + delta * np.random.uniform(size=self.dimension())

class RangeBearing:

    def __init__(self, sigma_range, sigma_bearing, pos):
        #self.sigma_range = sigma_range
        #self.sigma_bearing = sigma_bearing
        self.__R = np.diag([sigma_range**2, sigma_bearing**2])
        self.pos = np.array(pos)
        self._dist = lambda v: np.linalg.norm(v[0:2]-self.pos)

    def dimension(self):
        return 2

    def H(self, x):#[dR/dx,dR/dy,0,0;dB/dx,dB/dy,0,0]
        tmp = np.zeros(shape=(2,len(x)))
        tmp[:,0:2] = np.array([
            [ (x[0]-self.pos[0]), x[1]-self.pos[1]] / self._dist(x),#the der. of range
            [-(x[1]-self.pos[1]), x[0]-self.pos[0]] / (self._dist(x)**2)#the der. of bearing
        ])
        return tmp

    def R(self):
        return self.__R

    def h(self, x):
        return np.array([
            self._dist(x),
            np.arctan2(x[1]-self.pos[1], x[0]-self.pos[0])
        ])

    def measure(self, x):
        return np.random.multivariate_normal(self.h(x), self.__R)

    def sample(self, ranges):
        assert(len(ranges)==self.dimension())
        rand = np.random.uniform(size=self.dimension())
        R = ranges[0, 0:2]
        a = ranges[1, 0:2]
        return np.array([
            np.sqrt((R[1]**2-R[0]**2)*rand[0] + R[0]**2),
            a[0] + (a[1]-a[0])*rand[1]
        ])
if __name__ == '__main__':
    ATest = ConstantVelocity()
    xT = ATest.sample(np.array([[1,2],[3,4]]))
    print(xT)
    Btest = RangeBearing(1.,2.,[10.,20.])
