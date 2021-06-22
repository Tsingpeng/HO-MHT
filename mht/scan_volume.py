import numpy as np

from mht.utils.generation import measurements

class Volume(object):

    def __init__(self, P_D, clutter_lambda, init_lambda):
        self._pd = P_D#检测概率
        self._lambda_c = clutter_lambda#杂波的参数
        self._lambda_init = init_lambda#初始(birth)的参数

    def P_D(self):#返回检测概率
        return self._pd

    def _intensity(self, lam):#返回密度
        return lam/self.volume()

    def clutter_intensity(self, lambda_c=None):#杂波的密度(lambda_c可省略)
        return self._intensity(self._lambda_c if lambda_c is None else lambda_c)

    def initiation_intensity(self, lambda_init=None):#初始的密度(lambda_init可省略)
        return self._intensity(self._lambda_init if lambda_init is None else lambda_init)

    def scan(self, objects, measmodel, ranges):#返回位于测量体积内的量测zkk_1值传感器量测measure[0]
        assert(ranges.shape[0] == measmodel.dimension())
        objs_inside = {i: x for i, x in objects.items() if self.is_within(measmodel.h(x))}
        return measurements([objs_inside], measmodel, self._pd, self._lambda_c, ranges)[0]
    #虚函数,后面定义
    def volume(self):
        raise NotImplementedError()

    def is_within(self, z):
        raise NotImplementedError()

class CartesianVolume(Volume):

    def __init__(self, ranges, P_D, clutter_lambda, init_lambda):
        """
        ranges is [[x0_min, x0_max], [x1_min, x1_max], ...]
        """
        assert(ranges.shape[1]==2)
        assert((ranges[:,0] <= ranges[:,1]).all())#np.all()
        self._ranges = np.array(ranges)
        super(CartesianVolume, self).__init__(P_D, clutter_lambda, init_lambda)

    def volume(self):#得到ranges表示的矩形/长方体的面积/体积
        return (self._ranges[:,1] - self._ranges[:,0]).prod()#计算体积/面积

    def is_within(self, z):#给定的z是否位于volume中
        return ((self._ranges[:,0] <= z) & (z <= self._ranges[:,1])).all()

    def scan(self, objects, measmodel):
        return super(CartesianVolume, self).scan(objects, measmodel, self._ranges)