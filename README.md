# HO-MHT (Hypothesis Oriented Multi-Hypothesis Target Tracking/基于假设的多假设目标跟踪算法) 

语言：Python3.x 

外部依赖包：numpy, scipy, tqdm.

注：使用murty算法进行优化，其效果和TO-MHT类似


## 1. tracker.py
包含３个类(局部假设管理，航迹(假设簇)管理，全局假设管理)以及内部对数归一化函数．
### 1.1 _normalize_log_sum(items)
对给定的数组`items`进行对数归一化处理.
我们先得到`items`从大到小的索引序列,从而找到`items`中的最大值$w_0$,从而得到对数和为:
$$ls = w_0+ln[1+\sum_{i=1}^{n}exp(w_i-w_0)]$$
$$Norm = items - ls$$
其中$\{w_0,w_1,\cdots,w_n\}$为`items`的降序排列的元素,返回归一化值Norm和对数和ls.
### 1.2 LocalHypothesis类
用以产生局部假设,每创建一次实例就会分配一个`id`,创建时的初始化变量包括:target,LLR,LLR_max,loglikelihood.
#### 1.2.1 id()
返回局部假设的id
#### 1.2.2target()
深拷贝变量`target`
#### 1.2.3 density()
深拷贝变量`target.density()`
#### 1.2.4 predict(t_now)
返回当前的预测结果
#### 1.2.5 log_likelihood
返回对数似然
#### 1.2.6 is_dead
返回目标是否消失
#### 1.2.7 is_confirmed
返回目标是否确认
#### 1.2.8 new_from_hit(cls, self, z, hit_llhood, t_hit)类方法!!!
有量测时从新的潜在航迹得到新的似然比$L(k)=L(k-1)+\Delta L(k)$,返回新的局部假设
#### 1.2.9 new_from_miss(cls, self, miss_llhood, t_now)类方法!!!
没有量测时得到新的似然比$L(k)=L(k-1)+\Delta L(k)$,返回新的局部假设
### 1.3 Track类
航迹：假设的集合（位于椭圆门限内）
初始化变量有:lhyps&rarr;{id:object},trid&rarr;每创建一个实例,分配一个track_id.
#### 1.3.1 id
返回当前Track的id
#### １.3.2 __call__(lhyp_id)
像函数一样调用当前对象.如:A=Track(...);Track(lhyp_id)  
判断输入的hypo_id在lhyps中否,若在返回id对应的object,否则返回None  
#### 1.3.3 add(local_hypothesis)
添加假设local_hypothesis到lhyps中
#### 1.3.4 estimate(lhyp_id=None)
返回density:  
- 不指定假设的id,返回所有假设的density  
- 指定假设的id的话,返回这个id对应的假设的density  

#### 1.3.5 is_within(volume)
返回量测值的预测值是否在量测体积内
#### 1.3.6 log_likelihood_ratio(lhyp_id=None)
返回对数似然比:
- 未指定,返回所有的LLR
- 指定ID,返回指定ID的LLR

#### 1.3.7 log_likelihood_ratio_max(lhyp_id=None)
返回最大似然比:
- 未指定,返回所有的LLR
- 指定ID,返回指定ID的LLR

#### 1.3.8 log_likelihood(lhyp_id)
返回指定ID的似然
#### 1.3.9 dead_local_hyps
返回已消失的局部假设id列表
#### 1.3.10 confirmed_local_hyps
返回已确认的局部假设id列表
####　1.3.11 terminate(lhyp_ids)
更新假设:保留不在输入lhyp_idx中的假设,即lhyp_ids是终止假设的集合(列表)
#### 1.3.12 select(lhyp_ids)
更新假设:保留在输入lhyp_idx中的假设,即lhyp_ids是选择假设的集合(列表)
#### 1.3.13 predict(t_now)
预测:对每个假设进行预测
#### 1.3.14 update(Z, volume, t_now)
更新:更新局部假设（对于每个局部假设进行处理）  

- 得到在椭圆门内的量测和量测索引  
- 得到Density.predicted_likelihood  
- 得到有量测存在下的$\Delta L(k) = ln f(z,zkk_1,S)+ln(P_D/P_F)$  
- 未有量测时:$\Delta L(k) = ln(1-P_D)$  
- 更新局部假设new_lhyps[lid][ｊ]=LocalHypothesis,其中，lid表示假设编号，ｊ表示量测编号，即对应每个量测和上个假设产生的新假设集合  

### 1.4 Tracker类
跟踪器初始化变量：  
_M&rarr;murty算法中的m-best个数  
_weight_threshold&rarr;假设的权值阈值  
tracks&rarr;{Track_id:Track}  
ghyps&rarr;全局假设[{Track_ID:Hypo_ID},...{Track_ID:Hypo_ID}]  
gweights&rarr;这个权重是ｍ-best的权重，这里设定为０，即m-best的系数为$M\times e^{gweight}$  
#### 1.4.1 create_track_trees(detections, volume, targetmodel, t_now)
创建航迹树，对每个量测进行处理，得到相应的全局假设，新的航迹Track以及初始化成本  

- 初始化llr,密度  
- 对每个量测进行处理  
- 建立目标  
    -  生成新的假设    
    - 并由新生成的假设建立起新的Track  
    - 得到全局假设   
    - 保存得到的tracks.   
- 计算初始化成本  
- 返回全局假设和初始化成本  
####　1.4.2 _unnormalized_weight(ghyp)
返回新生成航迹所对应的假设的对数似然之和
#### 1.4.3 update_global_hypotheses(track_updates,Z,targetmodel,volume,M, weight_threshold,t_now)
更新全局假设:  
track_updates:对tracks中的每个在量测体积中的航迹进行更新
OrderedDict({trid:dict({lid:OederDict({z_idx:LocalHypothesis})})})  
```
        track_updates = OrderedDict([
            (trid, track.update(Z, volume, t_now))
            for trid, track in self.tracks.items() if track.is_within(volume)
        ])
```
Z:当前时刻的量测集合  
targetmodel:目标模型  
volume:scan_volume  
M:m-best matching  
weight_threshold: 权重的阈值   
t_now:现时刻  

- 如果当前航迹集不为空：   
    - 对每个global_hypo和track_update应用murty算法  
    - 若航迹参与分配问题(track in hyp is included in assignment problem)  
        - 由murty算法得到第`i`分配  
            - 对每个global_hypo进行处理：  
                - 若航迹id在分配中:  
                    - 更新tracks和new_ghyp  
                    - Track中添加lhyp,使用Track.add(lhyp)  
                - 若航迹id不在分配中：　　
                    - 更新new_ghyp,等待下次判断　　
            - 对于unsignment,生成新的航迹假设init_ghyp.(create_track_trees)　　
            - 使用字典的update方法将init_ghyp添加到new_ghyp中．　　
            - 由init_ghyp得到相应的$\Delta weight$.(_unnormalized_weight).  　　　
            - 更新权重($weight+\Delta weight$)集合new_weight和全局假设集合new_ghyp.    　　　
    - 若航迹不参与分配问题：  
        - 权重集合和全局假设集合保持原样(添加上一时刻集合)  
- 如果tracks为空,重新产生航迹和权值new_weight=>[double],newghyps=>[{Track_ID:Hypo_ID}]　　
- 注意检查（做出声明）：new_ghyps的一个dict应对应new_weights的一个double　　
- 去掉消失的航迹假设(prune_dead)并将权重对数归一化　　
- 假设剪枝(hypothesis_cap/依照阈值去除)并对数归一化权重　　
- 1-scan-pruning:k时刻出现的不确定性在k+1时刻解决　　
    - 更新track中的_hypo:只保留track和ghyp中lid相同的hypo
- 用得到的new_weight和new_ghyp替换self.gweights和self.ghyps

#### 1.4.4 prune_dead
剪枝(减去消失点):　　
weight:权重集=>[double...double];　　
global_hypotheses:[{Track_ID:Hypo_ID},...,{Track_ID:Hypo_ID}]　　
return: 返回剪完枝的权重(pruned_weights)和假设(pruned_ghyps)　　
dead_hypo指的是那些检测到的目标局部假设预测时间与更新时间差超过限度或历史航迹中有2个以下是T,认为它是要删去的假设．　　
#### 1.4.5 hypothesis_prune静态方法（可通过类名/实例名调用）
删除假设:权重大于阈值的保留,否则删除
#### 1.4.6 hypothesis_cap静态方法
capture hypothesis:保留前M个权值最大的假设
#### 1.4.7 estimates
当权重集不为0,若不止是输出ghpys中的确认航迹假设　或 最大权值对应的ghyp{trid:lid}中的lid是在tracks[trid]的确认假设id列表中,则输出这个航迹的对应假设的估计:{trid: Density}
#### 1.4.8 predict
对tracks中的每个track中的每个hypo进行预测  
#### 1.4.9 terminate_tracks
删除未使用的航迹(在tracks中删除{trid:lid})
#### 1.4.10 update(detections, volume, targetmodel, t_now)
track_updates:对tracks中的每一个航迹进行更新OrderedDict{trid:track_update}
并由此进行全局假设更新（update_global_hypotheses），并删除未使用的航迹（terminate_tracks）
#### 1.4.11 process(detections, volume, targetmodel, t_now)
由量测对航迹进行预测(predict),更新(update)
# 2. data_association.py
包含CostMatrix类，使用murty算法得到n-best assignment.
## 2.1 CostMatrix类
global_hypothesis:{trid:lid}   
track_updates:OrderedDict({trid:dict({lid:OederDict({z_idx:LocalHypothesis})})})   

- 初始化输入全局假设global_hypothesis和航迹的更新track_updates，并由此得到:    
    - _included_trids:track_updates中包含与global_hypothesis相同的trid列表．    
- 生成新的局部假设函数&rarr;计算非MISS似然函数&rarr;计算_inclueded中的所有值的非MISS似然(竖着堆叠起来)$<track \times localhyp>$   
- 计算所有MISS似然&rarr;得到所有的MISS似然&rarr;$<track \times track>$   
- 从而得到成本矩阵_Matrix为$<track \times localhyp+track>$  

### 2.1.1 tracks
返回_included_trids中的元素（用于判断是否有元素）
### 2.1.2 solutions
使用murty算法得到n-best的成本(未使用)，分配列表，未分配列表．其中索引为trackID,值为对应的localHypID.
# 3. gaussian.py
## 3.1 mahalanobis2(x, mu, inv_sigma)函数
求得马氏距离：应用在椭圆门中．
$$(x-\mu)^T  S^{-1} (x-\mu)$$
## 3.2 kalman_predict(density, motion, dt)函数
得到KF的预测(Density类)  
density:Density类的实例  
motion:motionModel   
dt:sampling time
## 3.3 kalman_update(density, z, inv_S, measure)函数
得到KF的更新(Density类)  
density:Density类实例   
z:$z_k$   
inv_S:$S^{-1}$   
measure:measureModel   
## 3.4 predicted_likelihood(density, z, S, measure)函数
预测对数似然$\Lambda(x_{k|k-1})=P(z_k|x_{k|k-1})=N(z_{k|k-1},S_k)=>ln f(z,z_{k|k-1},S)$
## 3.5 innovation(density, measure)函数
得到信息矩阵：
$$S_k=H_kP_{k|k-1}H_k^T+R_k$$

## 3.6Density类
包含计算Probability以及与Point Density相关的方法．包括KF,波门及其他．   
我们设置slots属性来限制class的属性仅为`x`和`P`   
初始化x和P为64位表示的矩阵．
### 3.6.1 __eq__
重载"=",同类型时判断是否相等返回T/F
### 3.6.2 cov_ellipse(measure=None, nstd=2)
由z,Pz得到椭圆信息(中心,长轴,短轴,方向)
### 3.6.3 ln_mvnpdf(x)
将正态分布取对数ln f(x,self.x,self.P)
### 3.6.4 gating(Z,measure,size2,inv_S=None,bool_index=False)
使用椭圆门得到在椭圆门内的量测值和量测索引
### 3.6.5 predicted_likelihood(z, measure, S=None)
预测对数似然$\Lambda(x_{k|k-1})=P(z_k|x_{k|k-1})=N(z_{k|k-1},S_k)=>ln f(z,z_{k|k-1},S)$
### 3.6.6 predict(motion, dt)
KF predict,预测self.x,self.P
### 3.6.7 update(z, measure, inv_S=None)
KF update,更新self.x,self.P
### 3.6.8 sample
从$x~N(self.x,self.P)$中采样
# 4. generation.py
产生真实值和量测值  
## 4.1 _ground_truth_fixed_step(t_length,x_birth,t_birth,t_death, motionmodel,dt)函数
依据模型，初始状态，以及出现/消失时刻生成一个包含状态字典的真实轨迹列表：   
trajectories:`trajs[timeK]={idx:state}`    
其中，    
t_length:int 仿真时间    
x_birth:list[int] 开始状态     
t_birth:list[int] 开始时间   
t_death:list[int] 消失时间   
motionmodel:运动学模型(状态方程)类   
dt:采样时间   
## 4.2 ground_truth(t_end, x_birth, t_birth, t_death, motionmodel, dt=1.0)
归一化到dt=1,再依据_ground_truth_fixed_step产生一个包含状态字典的真实轨迹列表
## 4.3 measurements(ground_truth, measmodel, P_D, lambda_c, range_c)
meas是一个二维列表，有多少行就表示有多长时间，每一行包含该时刻的量测值，以及杂波（产生时间服从泊松分布，空间服从均匀分布).    
range_c是均匀分布采样的范围`[[z0_min, z0_max], [z1_min, z1_max]`.   
## 4.4 random_ground_truth(t_end, init_state_density, init_lambda, P_survival, motionmodel, dt)
随机产生真实目标．目标产生由参数为`init_lambda`的泊松分布产生．   
目标的初始状态是从服从运动学模型的初始状态密度和它的轨迹中采样得出（只要它存在）．   
目标的消失由参数为`P_survival`均匀分布决定是否留下．    
返回一个包含状态字典的真实轨迹列表．（可以不使用这个函数，只用ground_truth即可）.   
# 5. motionmodel.py
运动学模型：NCV-2D,NT-2D(暂不使用)   
## 5.1 ConstantVelocity2D类
NCV-2D运动学模型．$\sigma$影响Q阵的大小．   
### 5.1.1 dimension
返回状态维度，这里使用NCV-2D,返回４.
### 5.1.2 F(x, dt=DT_DEFAULT)
这里返回Ｆ阵.
### 5.1.3 Q(dt=DT_DEFAULT)
这里返回Q阵.
### 5.1.4 f(x, dt=DT_DEFAULT)
返回$F\times x$.

# 6. measmodel.py
包括线性NCV-2D模型和二维主动式雷达量测模型．　　　
## 6.1 ConstantVelocity类
由$\sigma$得到R阵　　
### 6.1.2 dimension
NCV-2D的量测模型的维度为２．
### 6.1.3 H(x)
返回H矩阵
### 6.1.4 R
返回R阵
### 6.1.5 h(x)
返回$H\times x$
### 6.1.6 measure(x)
返回$H\times x + N(0,R)$
### 6.1.7 sample(ranges)
用于产生杂波．
## 6.2 RangeBearing类
由$\sigma_r$和$\sigma_b$产生相应的R阵．并得到雷达基站的位置．
### 6.2.1 dimension
二维雷达，返回２
### 6.2.2 H(x)
返回dR和dB的一阶导数`[dR/dx,dR/dy,0,0;dB/dx,dB/dy,0,0]`，即使用EKF进行处理．
### 6.2.3 R
返回R阵
### 6.2.4 h(x)
返回`[range, bearing]`
### 6.2.5 measure(x)
返回$h(x)+N(0,R)$.
### 6.2.6 sample(ranges)
用于产生杂波．
# 7.target.py
目标基类．用于其子类继承．
## 7.1 Target类
输入density, t_now．　　
_density:目标的density类，主要是该目标KF的预测更新以及波门．　　
_time:当前时刻（预测时刻）．　　
_time_hit:更新时刻．　　　
_hit_history:队列长度为５，这里保留最近的5个数据`[T,T,F,F,T]`,初始为`Ｔ`.    
### 7.1.1 predict
对目标进行KF预测
### 7.1.2 update_hit
对目标进行KF更新
### 7.1.3 update_miss
对None值(丢帧)进行更新，记录到_hit_history中．
### 7.1.4 is_confirmed
目标的确认：历史中有2个以上是T,认为它是存在的目标
### 7.1.5 is_dead
预测时间与更新时间差超过限度或目标历史中有2个以下是T,认为它是要删去的目标．
### 7.1.6 density
返回density类实例(for test&debug)
### 7.1.7 with_in
判断量测是否位于量测体积中  
### 7.1.8 from_one_detection
virtual  func. see below.
### 7.1.9 motion
virtual func.　see below.
### 7.1.10 gating
virtual func. see below.
### 7.1.11 predicted_likelihood
virtual func. see below.
### 7.1.12 max_coast_time
virtual func. see below.
# 8.cv_target.py
NCV-2D运动学模型相关
## 8.1 TargetPosition_CV2D(Target)子类
_motion,_measure由NCV-2D线性模型给定．   
椭圆门的门限由卡方分布给定（e.q. 9.2）.
### 8.1.1 _inv_h
由量测$ｚ$得到初始状态$x_0$   
### 8.1.2 _P0
得到初始协方差阵．　　
### 8.1.3 from_one_detection(cls, detection, t_now)
从一次检测中得到新的目标$x~N(_inv_h,P_0)$．　　
### 8.1.4 motion
返回当前运动学模型（类）
### 8.1.5 measure
返回当前量测方程（类）
### 8.1.6 gating
返回椭圆门内的量测值和索引．
### 8.1.7 predicted_likelihood
返回似然函数的预测值
### 8.1.8 max_coast_time
目标is_dead的时间差上限．

# 9. scan_volume.py
检测位于量测体积内的量测值
## 9.1 Volume基类
_pd：检测概率．　　　
_lambda_c: 杂波时间服从泊松分布的系数．　　　
_lambda_init:初始(birth)的参数．　　
### 9.1.1 P_D
返回检测概率．　　
### 9.1.2 _intensity
返回密度值．　　
### 9.1.3 clutter_intensity
返回杂波密度
### 9.1.4 initiation_intensity
返回初始(birth)密度`<用于初始化llr>`
### 9.1.5 scan(objects, measmodel, ranges)
返回在量测体积内的量测值->一维列表
### 9.1.6 volume
virtual func. see below.
### 9.1.7 is_within
virtual func. see below.
## 9.2 CartesianVolume(Volume)类
实现基于NCV-2D的Volume子类．　　　
### 9.2.1 volume
对于NCV线性量测模型，得到矩形面积Volume.
### 9.2.2 is_within
返回量测是否在矩形Volume中．
### 9.2.3 scan
直接调用父类scan方法．
# 10. murtyPy.py
P.S. 这个可以使用Ｃ++编写然后再用pybind11粘到Python3上加快处理速度．
这里我们使用`scipy.optimize`中的`linear_sum_assignment`函数进行处理．
注意，这里我们设定极大值$Inf=10^9$而不采用$Inf=float("Inf")$.
## 10.1 Hungarian(matrix)
使用linear_sum_assignment得到costMatrix的每一行分配结果（列表表示）和代价(cost).
## 10.2 MurtyPartition(N, a, type)
使用最小分配`a`得到分割节点`N`.     
N:非空的子集，包含所有的分割方案，N分为`Inclu`部分和`Exclu`部分.     
a:$nMeas\times 1$向量，最小分配.    
type: 0=>N-to-N assignments; 1=>M-to-N assignments.    
nodeLists:返回的分割列表．$所有分割的分配\cup a的分配＝＞所有的节点Ｎ的分配$;每个分割也表示为Inclu和Exclu部分．　　
## 10.3 murty(costMat,k)
使用murty算法即求在已知上一个最大图匹配时，我们得到次之的图匹配关系．返回costMat对应的前ｋ个最大图匹配以及相应的最小ｋ损失．
# 11. single_run.py
初始化参数：　　　　

- 使用模型为NCV-2D模型，线性量测.　　　
- 初始化k个目标的初始状态均值和初始协方差.　　
- 初始化检测体积范围.　　　
- 初始化杂波参数$\lambda_c$.　　
- 初始化生成新生目标参数$\lambda_i$.　　
- 初始化采样时间为１. 　　
- 初始化M.C.次数为１.　　
- 初始化CartesianVolume类实例.　　
- 初始化k个目标状态.　　


对于每次M.C.   

- 设置Tracker.  
- 生成一次M.C.的所有时序真实数据.　　
- 生成一次M.C.的所有时序量测数据.　　
- 对一次M.C.的每个时刻的量测进行处理跟踪.　　
# 12.constant
- EPS:极小的浮点正数.   
- LARGE:极大的浮点正数.　　　
- LOG_0:$ln(0)$使用极小的浮点负数表示.　　　
- MISS:None







