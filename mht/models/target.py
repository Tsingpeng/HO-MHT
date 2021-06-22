from collections import (deque)

class Target:

    def __init__(self, density, t_now):
        self._density = density
        self._time = t_now
        self._time_hit = t_now
        self._hit_history = deque(maxlen=5)#注意:这里保留最近的5个数据[T,T,F,F,T]
        self._hit_history.append(True)

    def predict(self, t_now):#航迹预测
        self._density.predict(self.motion(), dt=t_now-self._time)
        self._time = t_now

    def update_hit(self, detection, t_now):#航迹更新
        self._density.update(detection, self.measure())
        self._time_hit = t_now
        self._hit_history.append(True)

    def update_miss(self, t_now):#更新丢包
        self._hit_history.append(False)

    def is_confirmed(self):#历史航迹中有2个以上是T,认为它是存在的航迹
        return len(self._hit_history) > 2 and self._hit_history.count(True) >= 2

    def is_dead(self):#预测时间与更新时间差超过限度或历史航迹中有2个以下是T,认为它是要删去的航迹
        timeout = (self._time-self._time_hit) > self.max_coast_time()
        return timeout or (len(self._hit_history) > 2 and self._hit_history.count(True) < 2)

    def density(self):#返回density
        return self._density

    def is_within(self, volume):#判断是否位于量测体积中
        return volume.is_within(z = self.measure().h(self._density.x))
    #以下为虚函数,会在之后(example)中定义
    @classmethod
    def from_one_detection(cls, detection, t_now):
        raise NotImplementedError()

    @classmethod
    def motion(self):
        raise NotImplementedError()

    @classmethod
    def measure(self):
        raise NotImplementedError()

    def gating(self, detections):
        raise NotImplementedError()

    def predicted_likelihood(self, detection):
        raise NotImplementedError()

    def max_coast_time(self):
        raise NotImplementedError()
