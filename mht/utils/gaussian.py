import numpy as np

def mahalanobis2(x, mu, inv_sigma):
    """得到马氏距离"""
    d = x-mu
    return d.T @ inv_sigma @ d

# def moment_matching(log_w, densities):
#     """矩匹配,给定ln(w),density多个类实例"""
#     w = np.exp(log_w)#得到w = exp(ln(w))
#     x_weighted = np.dot(w, [d.x for d in densities])#求出w@x,即x的加权和
#     spread = lambda x, mu: (x-mu)[np.newaxis].T @ (x-mu)[np.newaxis]#得到矩阵spread
#     P_weighted = sum([w[i] * (d.P + spread(d.x, x_weighted)) for i,d in enumerate(densities)])
#     return Density(x_weighted, P_weighted)

def kalman_predict(density, motion, dt):
    """得到KF的预测(Density类):density:Density类的实例,motion:motionModel,dt:sampling time"""
    F = motion.F(density.x, dt)#F
    x = motion.f(density.x, dt)#xkk_1=F*xk_1
    P = F @ density.P @ F.T + motion.Q(dt)#Pkk_1 = F*Pk_1*F'+Q
    return Density(x=x, P=P)

def kalman_update(density, z, inv_S, measure):
    """得到KF的更新(Density类):density:Density类实例,z:zk,inv_S:S^{-1},measure:measureModel"""
    H = measure.H(density.x)#H
    K = density.P @ H.T @ inv_S#K = Pkk_1*H'*S^{-1}
    x = density.x + K @ (z-measure.h(density.x))#xk = xkk_1+K*(z-zkk_1)
    #P = density.P - (K @ H @ density.P)#Pk = Pkk_1-K*H*Pkk_1
    tmp = K @ H    #Joseph Form
    subArray = np.eye(*tmp.shape) - tmp
    P = subArray @ density.P @ subArray.T + K @ measure.R() @ K.T
    return Density(x=x, P=P)

def predicted_likelihood(density, z, S, measure):
    """预测对数似然\Lambda(x_{k|k_1})=P(zk|xkk_1)=N(zkk_1,Sk)=>ln f(z,zkk_1,S)"""
    zbar = measure.h(density.x)
    d = Density(x=zbar, P=S)
    return d.ln_mvnpdf(z)

def innovation(density, measure):
    """得到新息矩阵Sk=Hk*Pkk_1*Hk'+Rk"""
    H = measure.H(density.x)
    S = (H @ density.P @ H.T) + measure.R()
    S = 0.5 * (S + S.T) # Ensure positive definite
    return S

# def ellipsoidal_gating(density, Z, inv_S, measure, size2):
#     """椭球(椭圆)门(和Density.gating一样).使用椭圆门得到在椭圆门内的量测值和量测索引;
#     Z:量测集,measure:meauremodel;size2:门限大小"""
#     zbar = measure.h(density.x)
#     in_gate = np.array([mahalanobis2(zi, zbar, inv_S) < size2 for zi in Z])
#     return (Z[in_gate,:], in_gate)

class Density:
    """波门以及KF及其他"""
    __slots__ = ('x', 'P')#快速get/set属性x(xkk_1或xk),P(Pkk_1或Pk)

    def __init__(self, x, P):
        """先将x,P转为float64表示是的矩阵"""
        self.x = np.float64(np.array(x))
        self.P = np.float64(np.array(P))

    def __repr__(self):
        """打印x"""
        return "<density x={0}>".format(self.x)

    def __eq__(self, other):
        """重载"=",同类型时判断是否相等返回T/F"""
        if isinstance(other, Density):#判断other是否是Density类型
            return np.array_equal(self.x, other.x) and np.array_equal(self.P, other.P)
        return NotImplemented

    def cov_ellipse(self, measure=None, nstd=2):
        """由z,Pz得到椭圆信息(中心,长轴,短轴,方向)"""
        if measure is not None:#从measure中得到量测协方差和量测
            H = measure.H(self.x)
            Pz = H @ self.P @ H.T
            z = measure.h(self.x)
        else:#得到量测协方差和量测(2维)
            Pz = self.P[0:2,0:2]
            z = self.x[0:2]
        #从Pz中得到相应的(长轴,短轴,方向)
        eigvals, vecs = np.linalg.eigh(Pz)#计算Pz的特征值,特征向量
        order = eigvals.argsort()[::-1]#特征值降序排列的索引位置,如eigval=[0.2,0.1,0.3],得到order=[2,0,1]
        eigvals, vecs = eigvals[order], vecs[:,order]#重排特征值\特征向量
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))#弧度转角度
        r1, r2 = nstd * np.sqrt(eigvals)
        #加上z,得到椭圆信息(中心,长轴,短轴,方向)
        return z, r1, r2, theta

    def ln_mvnpdf(self, x):
        """对正态分布取对数,ln f(x,self.x,self.P)"""
        ln_det_sigma = np.log(np.linalg.det(self.P))#ln(|P|)
        inv_sigma = np.linalg.inv(self.P)#P^{-1}
        return -0.5 * (ln_det_sigma + mahalanobis2(np.array(x), self.x, inv_sigma) + len(x)*np.log(2*np.pi))

    def gating(self, Z, measure, size2, inv_S=None, bool_index=False):
        """使用椭圆门得到在椭圆门内的量测值和量测索引;Z:量测集,measure:meauremodel;size2:门限大小"""
        if inv_S is None:
            inv_S = np.linalg.inv(innovation(self, measure))

        zbar = measure.h(self.x)#zbar=H*x
        is_inside = lambda z: mahalanobis2(z, zbar, inv_S) < size2#位于门内(z-Hx)'S^{-1}(z-Hx)<size2
        if bool_index:#bool_index==True:得到在每个量测是否在门内的T/F
            in_gate = np.array([is_inside(z) for z in Z])
        else:#bool_index==True:得到在门内的量测的索引
            in_gate = np.array([i for i, z in enumerate(Z) if is_inside(z)], dtype=int)

#        if len(in_gate) > 0:
        return (Z[in_gate], in_gate)
#        else:
#            return (np.array([]), np.array([]))

    def predicted_likelihood(self, z, measure, S=None):
        """预测对数似然\Lambda(xkk_1)=P(zk|xkk_1)=N(zkk_1,Sk)=>ln f(z,zkk_1,S)"""
        zbar = measure.h(self.x)#H*x
        d = Density(x=zbar, P=innovation(self, measure) if S is None else S)
        return d.ln_mvnpdf(z)

    def predict(self, motion, dt):
        """KF predict,更新self.x,self.P"""
        predicted = kalman_predict(self, motion, dt)
        self.x, self.P = predicted.x, predicted.P
        return self

    def update(self, z, measure, inv_S=None):
        """KF update,更新self.x,self.P"""
        if inv_S is None:#S^{-1}
            inv_S = np.linalg.inv(innovation(self, measure))
        #更新xk,Pk
        updated = kalman_update(self, np.array(z), inv_S, measure)
        self.x, self.P = updated.x, updated.P
        return self

    # def kalman_step(self, z, dt, motion, measure):
    #     """KF's one-step predict: (xk_1,Pk_1)=>(xk,Pk)"""
    #     for zi in np.array(z): #some bugs here, Stop using it!!!
    #         self.predict(motion, dt).update(zi, measure) #maybe zi.T
    #
    #     return self

    def sample(self):
        """x~N(self.x,self.P)"""
        return np.random.multivariate_normal(self.x, self.P)

# class Mixture(object):
#     """ln归一化权值再矩匹配"""
#     def __init__(self, weights, components=[]):
#         self.weights = np.array(weights)#权值向量
#         self.components = np.array(components)#Density多个实例
#
#     def normalize_log_weights(self):
#         """对weight进行ln归一化=>weights = weights - (w0+ln[1+\sum_{1}^{n}exp(wi-w0)]),其中{w0,w1,...,wn}为降序排列的权值"""
#         if len(self.weights) == 1:
#             log_sum_w = self.weights[0]
#         else:
#             i = sorted(range(len(self.weights)), key=lambda k: self.weights[k], reverse=True)#i:降序索引,即得到weights由大到小排列的索引
#             max_log_w = self.weights[i[0]]#最大的权重
#             log_sum_w = max_log_w + np.log(1.0+sum(np.exp(self.weights[i[1:]]-max_log_w)))
#
#         self.weights -= log_sum_w
#
#         return log_sum_w
#
#     def moment_matching(self):
#         """ln归一化后的矩匹配,使用矩匹配的结果更新Density的x,P"""
#         self.normalize_log_weights()#对weight进行ln归一化
#         self.components = np.array([moment_matching(self.weights, self.components)])
#         self.weights = np.zeros(1)
#         return Density(x=self.components[0].x, P=self.components[0].P)