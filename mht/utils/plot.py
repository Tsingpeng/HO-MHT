import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from .gaussian import (Density)
""" Common Func. """
def to_state(obj):#obtain the state from Density.x
    if isinstance(obj, Density):
        return obj.x
    else:
        return obj

def show():#show the fig.
    plt.show()

def legend(x):#show the legend
    plt.legend(x)

class Plotter(object):
    """
    Plot the fig.
    to_plot_coordinates:get the function h(x)
    **kwargs: get the figure's setting
    """
    def __init__(self, to_plot_coordinates, **kwargs):
        self._to_z = to_plot_coordinates
        self._fig = plt.figure(**kwargs)

    def show(self):
        show()

    def legend(self,x):
        legend(x)

    def _project(self, objects, h):#get the [{id:h(x)}]
        """
        objects:list(dict(trid:stat))   h: func. h(x)
        return: list(dict(trid:h(stat)))
        """
        return [
            {i: h(to_state(x)) for i, x in objs.items()}
            if isinstance(objs, dict) else
            {i: h(to_state(x)) for i, x in enumerate(objs)}
            for objs in objects
        ]

    def _trajectory(self, objects_meas, **kwargs):
        """
        object_meas:list(dict(trid:z))
        """
        obj_index_to_id = list(set([ids for objs in objects_meas for ids in objs.keys()]))#得到所有的不重复的ids
        id_to_obj_index = {id: i for i, id in enumerate(obj_index_to_id)}

        n_objs = len(obj_index_to_id)
        t_length = len(objects_meas)
        #2-D measurement : (x,y). Here, zx=>z[time][index]=x  zy=>z[time][index]=y
        #zx/zy:  row:time  col:n_objects
        zx = np.full((t_length, n_objs), np.nan)
        zy = np.full(zx.shape, np.nan)

        for t, objs in enumerate(objects_meas):
            for id, z in objs.items():
                i = id_to_obj_index[id]
                [zx[t,i], zy[t,i]] = z

        self._fig.gca().plot(zx, zy, **kwargs)#画出同一目标在不同时刻的轨迹(paint in colume)
        return self._fig.gca().plot(zx, zy, **kwargs)# add this line to get  the ani.

    def trajectory_2d(self, objects, measmodel=None, **kwargs):
        h = self._to_z if measmodel is None else measmodel.h
        objects_meas = self._project(objects, h)
        self._trajectory(objects_meas, **kwargs)
        return self._trajectory(objects_meas, **kwargs)# add this line for ani.

    def measurements_2d(self, detections, inv_h=None, **kwargs):
        """setting inv_h or give the measurement[:2] as inv_h"""
        if inv_h is None:
            objects = self._project(detections, lambda z: z[:2])
        else:
            objects = self._project(self._project(detections, inv_h), self._to_z)

        self._trajectory(objects, linestyle='', **kwargs)
        return self._trajectory(objects, linestyle='', **kwargs)# add this line for ani

    def covariance_ellipse_2d(self, density, measmodel, nstd=2, **kwargs):
        z, r1, r2, theta = density.cov_ellipse(measmodel, nstd)
        ellip = Ellipse(xy=z, width=2*r1, height=2*r2, angle=theta, **kwargs)
        ellip.set_alpha(0.3)#设置透明度  0透明,1不透明
        self._fig.gca().add_artist(ellip)

        return ellip

    def covariances_2d(self, objects, measmodel=None, nstd=2, **kwargs):
        """objects: list(dict(trid:Density))"""
        for objs in objects:
            for obj in objs.values():
                self.covariance_ellipse_2d(obj, measmodel, nstd, **kwargs)
