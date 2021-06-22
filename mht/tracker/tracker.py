import numpy as np

from mht.constants import (EPS, LARGE, LOG_0, MISS)
from . import data_association

from collections import (OrderedDict)
from copy import (deepcopy)
"""The Core Code of MHT"""
def _normalize_log_sum(items):
    """对items进行ln归一化=>Norm = item - (w0+ln[1+\sum_{1}^{n}exp(wi-w0)]),
    其中{w0,w1,...,wn}为降序排列的权值,返回归一化值(item-log_sum)和对数和(log_sum)"""
    if len(items) == 0:#特判1
        return (items, None)
    elif len(items) == 1:#特判2
        log_sum = items[0]
    else:#和gaussian.Mixture.normalize_log_weights一样
        i = sorted(range(len(items)), key=lambda k: items[k], reverse=True)
        max_log_w = items[i[0]]
        log_sum = max_log_w + np.log(1.0+sum(np.exp(items[i[1:]]-max_log_w)))

    return (items-log_sum, log_sum)

class LocalHypothesis:

    def __init__(self, target, LLR, log_likelihood, LLR_max=None):
        self._target = target
        self._llr = LLR
        self._llr_max = LLR if LLR_max is None else LLR_max
        self._llhood = log_likelihood
        self._lid = self.__class__._counter#每创建一次实例就分配id(id加1)
        self.__class__._counter += 1#下次创建实例的id加1

    def id(self):#返回id
        return self._lid

    def target(self):#深拷贝target
        return deepcopy(self._target)

    def density(self):#深拷贝target.density
        return deepcopy(self._target.density())

    def predict(self, t_now):#返回预测结果
        self._target.predict(t_now)

    def log_likelihood_ratio(self):#返回对数似然比
        return self._llr

    def log_likelihood(self):#返回对数似然
        return self._llhood

    def is_dead(self):#返回目标是否消失
        return self._target.is_dead()

    def is_confirmed(self):#返回目标是否是确认目标
        return self._target.is_confirmed()
    #下面声明类方法(不用self)
    @classmethod
    def new_from_hit(cls, self, z, hit_llhood, t_hit):
        #由量测时从新的潜在航迹得到新的似然比L(k)=L(k-1)+\Delta L(k)
        target = deepcopy(self._target)
        target.update_hit(z, t_hit)
        llr = self._llr + hit_llhood
        return cls(
            target = target,
            LLR = llr,
            log_likelihood = hit_llhood,
            LLR_max = max(self._llr_max, llr),
        )

    @classmethod
    def new_from_miss(cls, self, miss_llhood, t_now):#没有量测时得到新的似然比L(k)=L(k-1)+\Delta L(k)
        target = deepcopy(self._target)
        target.update_miss(t_now)
        llr = self._llr + miss_llhood
        return cls(
            target = target,
            LLR = self._llr + miss_llhood,
            log_likelihood = miss_llhood,
            LLR_max = max(self._llr_max, llr)
        )

    def __repr__(self):
        return "<loc_hyp {0}: {1}>".format(self.id(), self.density())

LocalHypothesis._counter = 0#每创建一次实例_counter加1以得到唯一的id,初始id为0

class Track:

    def __init__(self, local_hypothesis):
        self._lhyps = {local_hypothesis.id(): local_hypothesis}#{id:object}
        self._trid = self.__class__._counter#每创建一次实例_counter加1以得到唯一的id
        self.__class__._counter += 1

    def __repr__(self):
        return "<track {0}: {1}>".format(self.id(), self._lhyps)

    def id(self):#当前Track的id
        return self._trid

    def __call__(self, lhyp_id):#像函数一样调用当前对象.如:A=Track(...);Track(lhyp_id)
        """判断输入的hypo_id在_lhyps中否,若在返回id对应的object,否则返回None"""
        if lhyp_id in self._lhyps.keys():
            return self._lhyps[lhyp_id]
        else:
            return None

    def add(self, local_hypothesis):#添加假设到self._lhyps中
        self._lhyps[local_hypothesis.id()] = local_hypothesis

    def estimate(self, lhyp_id=None):#返回density
        if lhyp_id is None:#不指定假设的id,返回所有假设的density
            return [lhyp.density() for lhyp in self._lhyps.values()]
        else:#指定假设的id的话,返回这个id对应的假设的density
            return self._lhyps[lhyp_id].density()

    def is_within(self, volume):#松弛了下,只要有一个量测点的预测值在volume中则返回T
        # Margin using covariance?
        return np.array([lhyp.target().is_within(volume) for lhyp in self._lhyps.values()]).any()

    def log_likelihood_ratio(self, lhyp_id=None):#对数似然比
        if lhyp_id is None:#未指定,返回所有的LLR
            return [lhyp.log_likelihood_ratio() for lhyp in self._lhyps.values()]
        else:#指定ID,返回制定ID的LLR
            return self._lhyps[lhyp_id].log_likelihood_ratio()

    # def log_likelihood_ratio_max(self, lhyp_id=None):#最大似然比
    #     if lhyp_id is None:
    #         return [lhyp._llr_max for lhyp in self._lhyps.values()]
    #     else:
    #         return self._lhyps[lhyp_id]._llr_max

    def log_likelihood(self, lhyp_id):#似然
        return self._lhyps[lhyp_id].log_likelihood()

    def dead_local_hyps(self):#消失的局部假设(返回假设id)
        return [lid for lid, lhyp in self._lhyps.items() if lhyp.is_dead()]

    def confirmed_local_hyps(self):#已确认的局部假设(返回假设id)
        return [lid for lid, lhyp in self._lhyps.items() if lhyp.is_confirmed()]

    # def terminate(self, lhyp_ids):#更新假设:保留不在输入lhyp_idx中的假设,即lhyp_ids是终止假设的集合(列表)
    #     self._lhyps = {
    #         lid: lhyp for lid, lhyp in self._lhyps.items()
    #         if lid not in lhyp_ids
    #     }

    def select(self, lhyp_ids):#更新假设,保留在输入lhyp_idx中的假设,即lhyp_ids是选择假设的集合(列表)
        self._lhyps = {
            lid: lhyp for lid, lhyp in self._lhyps.items()
            if lid in lhyp_ids
        }

    def predict(self, t_now):#预测:对每个假设进行预测
        for lhyp in self._lhyps.values():
            lhyp.predict(t_now)

    def update(self, Z, volume, t_now):#更新:更新LLR以及得到new_localhypo
        new_lhyps = dict()#new_lhyps[lid][t]=LocalHypothesis

        for lid, lhyp in self._lhyps.items():
            target = lhyp.target()
            (z_in_gate, in_gate_indices) = target.gating(Z)#得到在椭圆门内的量测和量测索引
            lh = np.array([target.predicted_likelihood(z) for z in z_in_gate])#得到Density.predicted_likelihood
            lhood = np.full(len(Z), LOG_0)#得到有量测存在下的\Delta L(k) = ln f(z,zkk_1,S)+ln(P_D/P_F)
            lhood[in_gate_indices] = lh + np.log(volume.P_D()+EPS) - np.log(volume.clutter_intensity()+EPS)
            new_lhyps[lid] = OrderedDict([
                (j, LocalHypothesis.new_from_hit(lhyp, Z[j], lhood[j], t_now)
                    if j in in_gate_indices else
                    None)
                for j in range(len(Z))
            ])
            P_G = 1.0#未有量测时:\Delta L(k) = ln(1-P_D)
            new_lhyps[lid][MISS] = LocalHypothesis.new_from_miss(lhyp, np.log(1.0 - volume.P_D()*P_G + EPS), t_now)

        return new_lhyps

Track._counter = 0#每次生成一个实例,id加1

class Tracker:

    def __init__(self, max_nof_hyps, hyp_weight_threshold):
        self._M = max_nof_hyps#m-best matching
        self._weight_threshold = hyp_weight_threshold#假设的权重阈值
        self.tracks = dict()#{Track_id:Track}
        self.ghyps = [dict()]#[{Track_ID:Hypo_ID},...{Track_ID:Hypo_ID}]
        self.gweights = np.array([np.log(1.0)])

    def create_track_trees(self, detections, volume, targetmodel, t_now):
        """
        return:
        new_ghyp=>新的航迹对应的假设{Track_ID:Hypo_ID}
        total_init_cost:初始化总成本
        """
        intensity_c = volume.clutter_intensity()#杂波密度
        intensity_new = volume.initiation_intensity()#初始密度
        llr0 = np.log(intensity_new+EPS) - np.log(intensity_c+EPS)#初始化llr

        new_ghyp = dict()#{Track_ID:Hypo_ID}
        for z in detections:#对同时刻的每一个量测处理
            llhood = np.log(volume.P_D()+EPS) - np.log(intensity_c+EPS)#\Delta L
            target = targetmodel.from_one_detection(z, t_now)#建立目标
            new_lhyp = LocalHypothesis(target, llr0, llhood)#产生新的假设 t_now
            new_track = Track(new_lhyp)#建立新的Track

            self.tracks[new_track.id()] = new_track#保存Track

            new_ghyp[new_track.id()] = new_lhyp.id()#新的航迹对应的假设

        total_init_cost = len(new_ghyp) * -llr0#初始化成本

        return new_ghyp, total_init_cost

    def _unnormalized_weight(self, ghyp):
        """
        ghyp:{Track_ID:Hypo_ID}
        return:新生成航迹所对应的假设的对数似然之和
        """
        return sum([self.tracks[trid].log_likelihood(lid) for trid, lid in ghyp.items()])

    def update_global_hypotheses(self, track_updates, Z, targetmodel, volume, M, weight_threshold, t_now):
        """
        更新全局假设:
        track_updates:对tracks中的每个航迹进行更新OrderedDict({trid:dict({lid:OederDict({z_idx:LocalHypothesis})})})
        Z:当前时刻的量测集合  targetmodel:目标模型  volume:scan_volume  M:m-best matching
        weight_threshold: 权重的阈值  t_now:现时刻
        return:
        """
        new_weights = list()#新的权重列表
        new_ghyps = list()#新的假设列表

        if self.tracks:#如果tracks不为空

            for ghyp, weight in zip(self.ghyps, self.gweights):#对每个global_hypo和track_update应用murty算法
                cost_matrix = data_association.CostMatrix(ghyp, track_updates)

                if cost_matrix.tracks():#track in hyp is included in assignment problem
                    # calcu. the n-best assignment,here we use `M` as the init_weight where we can get changes with time goes on
                    nof_best = np.ceil(np.exp(weight) * M)

                    for _, assignment, unassigned_detections in cost_matrix.solutions(nof_best):
                        new_ghyp = dict()
                        for trid, lid in ghyp.items():
                            if trid in assignment.keys():#若航迹id在分配索引列表中,更新tracks和new_ghyp
                                lhyps_from_gates = track_updates[trid][lid]#OrderDict({z_idx:LocalHypothesis})
                                detection = assignment[trid]#detection_id
                                lhyp = lhyps_from_gates[detection]#LocalHypothesis
                                new_ghyp[trid] = lhyp.id()
                                self.tracks[trid].add(lhyp)#Track中添加lhyp,使用Track.add(lhyp)
                            else:
                                # Not part of assignment problem, keep the old
                                new_ghyp[trid] = lid
                        #对于unsignment,生成新的航迹假设
                        init_ghyp, _ = \
                            self.create_track_trees(Z[unassigned_detections], volume, targetmodel, t_now)
                        #更新new_ghyp
                        new_ghyp.update(init_ghyp)#Dict's update method to add init_ghyp to new_ghyp
                        #得到相应的\Delta weight
                        weight_delta = self._unnormalized_weight(new_ghyp)
                        #更新权重集合和假设集合
                        new_weights.append(weight + weight_delta)
                        new_ghyps.append(new_ghyp)
                else:
                    # No track in hyp is included in assignment problem, keep the old
                    new_weights.append(weight)
                    new_ghyps.append(ghyp)

        else:#如果tracks为空,重新产生航迹和权值new_weight=>[double],newGhyps=>[{Track_ID:Hypo_ID}]
            init_ghyp, _ = self.create_track_trees(Z, volume, targetmodel, t_now)

            new_weights = [self._unnormalized_weight(init_ghyp)]
            new_ghyps = [init_ghyp]
        #new_ghyps的一个dict()对应new_weights的一个double
        assert(len(new_ghyps)==len(new_weights))
        #去掉消失的航迹假设并对数归一化权重(去除dead_hypo)
        new_weights, new_ghyps = self.prune_dead(np.array(new_weights), np.array(new_ghyps))
        new_weights, _ = _normalize_log_sum(new_weights)

        assert(len(new_ghyps)==len(new_weights))
        #假设剪枝并归一化权重(依照阈值去除)
        new_weights, new_ghyps = self.hypothesis_prune(new_weights, new_ghyps, weight_threshold)
        new_weights, _ = _normalize_log_sum(new_weights)
        #捕获假设并归一化权重(保留M个权重最大的航迹假设)
        new_weights, new_ghyps = self.hypothesis_cap(new_weights, new_ghyps, M)
        new_weights, _ = _normalize_log_sum(new_weights)

        # Kind of 1-scan MHT pruning...
        #1-scan-pruning:k时刻出现的不确定性在k+1时刻解决
        for trid, track in self.tracks.items():#更新track中的_hypo:只保留track和ghyp中lid相同的hypo
            track.select([ghyp[trid] for ghyp in new_ghyps if trid in ghyp.keys()])

        # Remove duplicate global hyps
#        hashable_ghyps = [tuple(d.items()) for d in new_ghyps]
#        unique_index = [hashable_ghyps.index(d) for d in set(hashable_ghyps)]
        # update the gw. and gh.
        self.gweights = new_weights
        self.ghyps = new_ghyps

    def prune_dead(self, weights, global_hypotheses):
        """
        剪枝(减去消失点): weight:权重集=>[double...double];
             global_hypotheses:[{Track_ID:Hypo_ID},...,{Track_ID:Hypo_ID}]
        return: 返回剪完枝的权重(pruned_weights)和假设(pruned_ghyps)
        """
        #dead_lhyps=>{trid:[lid,...,lid]}
        dead_lhyps = {trid: track.dead_local_hyps() for trid, track in self.tracks.items()}

        pruned_weights = list()
        pruned_ghyps = list()
        for weight, ghyp in zip(weights, global_hypotheses):
            w_diff = 0.0
            pruned_ghyp = dict()
            for trid, lid in ghyp.items():
                if lid in dead_lhyps[trid]:
                    w_diff += self.tracks[trid].log_likelihood(lid)
                else:
                    pruned_ghyp[trid] = lid

            if pruned_ghyp:
                pruned_weights.append(weight - w_diff)
                pruned_ghyps.append(pruned_ghyp)

        assert(len(pruned_weights)==len(pruned_ghyps))

        return np.array(pruned_weights), np.array(pruned_ghyps)

    @staticmethod
    def hypothesis_prune(weights, hypotheses, threshold):
        """删除假设:权重大于阈值的保留,否则删除"""
        keep = weights >= threshold
        return (weights[keep], hypotheses[keep])

    @staticmethod
    def hypothesis_cap(weights, hypotheses, M):
        """capture hypothesis:保留前M个权值最大的假设"""
        if len(weights) > M:
            i = np.argsort(weights)
            m_largest = i[::-1][:M]
            return (weights[m_largest], hypotheses[m_largest])
        else:
            return (weights, hypotheses)

    def estimates(self, only_confirmed=True):
        """return: 当权重集不为0,若不止是输出ghpys中包含的trid/lid,
        或 最大权值对应的ghyp{trid:lid}中的lid是在tracks[trid]的确认假设id列表中,
        输出:{trid: Density}"""
        if len(self.gweights) > 0:#权重集不为0,
            index_max = np.argmax(self.gweights)#取出最大权值的索引
            return {
                trid: self.tracks[trid].estimate(lid)
                for trid, lid in self.ghyps[index_max].items()
                if not only_confirmed or lid in self.tracks[trid].confirmed_local_hyps()
            }#返回{trid:tracks[trid].estimate(lid)}
        else:
            return {}

    def predict(self, t_now):
        """对tracks中的每个track中的每个hypo进行预测"""
        for track in self.tracks.values():
            track.predict(t_now)

    # def calculate_weights(self, global_hypotheses):
    #     """对global hypothesis求对数归一化的权值"""
    #     weights_updated = [
    #         sum([self.tracks[trid].log_likelihood(lid) for trid, lid in ghyp.items()])
    #         for ghyp in global_hypotheses
    #     ]#和``_unnormalized_weight``一样计算权值(加和)
    #     gweights, _ = _normalize_log_sum(np.array(weights_updated))#对数归一化
    #     return gweights

    def terminate_tracks(self):
        """删除未使用的航迹(在tracks中删除{trid:lid})"""
        trids_in_ghyps = set([trid for ghyp in self.ghyps for trid in ghyp.keys()])
        unused_tracks = set(self.tracks.keys()) - trids_in_ghyps
        for trid in unused_tracks:
            del self.tracks[trid]

    def update(self, detections, volume, targetmodel, t_now):
        Z = np.array(detections)
        #track_updates:对tracks中的每一个航迹进行更新OrderedDict{trid:track_update}
        track_updates = OrderedDict([
            (trid, track.update(Z, volume, t_now))
            for trid, track in self.tracks.items() if track.is_within(volume)
        ])
        #全局假设更新
        self.update_global_hypotheses(track_updates, Z, targetmodel, volume, self._M, self._weight_threshold, t_now)
        #删除未使用的航迹
        self.terminate_tracks()

    def process(self, detections, volume, targetmodel, t_now):
        """对航迹进行预测/跟踪,返回{trid: Density}"""
        self.predict(t_now)
        self.update(detections, volume, targetmodel, t_now)
        return self.estimates()

    # def debug_print(self, t):
    #     pass
    #     print("[t = {}]".format(t))
    #     #print(len(self.gweights))
    #     #print("Weights =")
    #     print(self.gweights)
    #     print(self.ghyps)
    #
    #     for trid in self.estimates().keys():
    #         print("    Track {} LLR = {} ({})".format(
    #            self.tracks[trid],
    #            self.tracks[trid].log_likelihood_ratio(),
    #            self.tracks[trid].log_likelihood_ratio_max()
    #         ))
    #
    #     print("")
