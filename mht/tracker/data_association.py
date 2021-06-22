import numpy as np

from mht.constants import (LOG_0, MISS)

from murty.murtyPy import murty
"""由global_hypothesis, track_updates得到成本矩阵继而得到m-best分配,每次返回"""
class CostMatrix:#m-best MHT中的假设矩阵用成本矩阵表示
    """
    global_hypothesis:{trid:lid}
    track_updates:OrderedDict({trid:dict({lid:OederDict({z_idx:LocalHypothesis})})})
    """
    def __init__(self, global_hypothesis, track_updates):
        self._included_trids = [trid for trid in track_updates.keys() if trid in global_hypothesis.keys()]
        #_included_trids保存track_updates和global_hypothesis共有的键
        if len(self._included_trids)==0:#若为空,_matrix=np.array([])
            self._matrix = np.empty(shape=(0,0))
            return
        #生成新的假设
        new_lhyps = lambda trid: track_updates[trid][global_hypothesis[trid]]
        #计算非MISS似然
        hit_likelihoods = lambda trid: np.array([
            LOG_0 if lhyp is None else lhyp.log_likelihood()
            for detection, lhyp in new_lhyps(trid).items() if detection is not MISS
        ])
        # 计算(count)_inclueded中的所有值的非MISS似然(竖着堆叠起来)<track x localhyp>
        c_track_detection = np.vstack(tuple(
            (hit_likelihoods(trid) for trid in self._included_trids)
        ))
        # 计算所有MISS似然
        miss_likelihood = np.array([
            new_lhyps(trid)[MISS].log_likelihood() for trid in self._included_trids
        ])
        # c_miss是值为ln_0的len(miss_likelihood)行len(miss_likelihood)列满矩阵,再用miss_likelihood填充主对角线
        c_miss = np.full(2*(len(miss_likelihood),), LOG_0)
        np.fill_diagonal(c_miss, miss_likelihood)# 用miss_likelihood填充c_miss的主对角线
        # c_track_detection:<tracks x localhpys>, c_miss:<miss_tracks x miss_tracks>
        self._matrix = -1.0 * np.hstack((c_track_detection, c_miss))#得到_matrix 成本矩阵

    def tracks(self):
        return self._included_trids[:]

    def solutions(self, max_nof_solutions):
        if not self._matrix.size:#特判
            return None
        #使用Murty算法,返回murty_solver对象
        #murty_solver = Murty(self._matrix)

        # Get back trid and detection nr from matrix indices
        to_trid = lambda t: self._included_trids[t]
        #得到max_nof_solutions (m) bests 分配
        # for _ in range(int(max_nof_solutions)):
        #     is_ok, sum_cost, track_to_det = murty_solver.draw()
        for track_to_det,sum_cost in murty(self._matrix,int(max_nof_solutions)):
            track_to_det = track_to_det.tolist()#Here, the index stands for track(row) and det stand for detection(meas/col)
            #is_ok:murty算法能否找到第i优分配(is_ok),分配的总成本(sum_cost),以及索引和分配(track_to_detection)
            # if not is_ok:#没有最优分配,返回None
            #     return None
            #n:miss;m:c_track_detection
            n, m_plus_n = self._matrix.shape
            #得到最优的分配{Track_id:Detection_ID}
            assignments = {
                to_trid(track_index): det_index if det_index in range(m_plus_n - n) else MISS
                for track_index, det_index in enumerate(track_to_det)
            }
            #没有分配到的索引[Detection_ID]
            unassigned_detections = [
                det_index for det_index in range(m_plus_n - n)
                if det_index not in track_to_det
            ]
            #iter一下
            yield sum_cost, assignments, np.array(unassigned_detections, dtype=int)

    def __repr__(self):
        return str(self._matrix)
