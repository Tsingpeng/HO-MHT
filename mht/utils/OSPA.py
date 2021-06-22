import numpy as np
from scipy.optimize import linear_sum_assignment
class OSPA:
    def __init__(self,X,Y,c,p,dim=4):
        """
        X:ground_truth:dict(trid:np.array([px,py,vx,vy])])
        Y:estimate_state:dict(trid:np.array([px,py,vx,vy]))
        c:cut-off parameter
        p:p parameter for metric
        dim: state's dimension of X and Y
        """
        self.c = c
        self.p = p
        # self.X = np.array([])
        # self.Y = np.array([])
        # get the matrix whose col. stand for state
        tmpX = []
        for val in X.values():
            val = val.tolist()
            if val:
                tmpX.append(val[:2])
        self.X = np.array(tmpX).T
        tmpY = []
        for val in Y.values():
            val = val.tolist()
            if val:
                tmpY.append(val[:2])
        self.Y = np.array(tmpY).T

    def _OPSA_fun(self,X,Y,c,p):
        """
        The  OPSA metric between X and Y
        X,Y: matrices of column vectors: Here exits n cols stand for n tracks
        c: cut-off parameter
        p: p-parameter for the metric
        return: scalar distance b/w X and Y
        """
        if not X.tolist() and not Y.tolist(): return 0#Special case
        if not X.tolist() or not Y.tolist(): return c
        # Calculate sizes of the input point patterns, i.e. how many tracks at this time
        n = X.shape[1]#size(X,2)
        m = Y.shape[1]#size(Y,2)

        # Calculate cost/weight matrix for pairinfs - fast method with vectorization
        # XX = np.hstack([X for _ in range(m)])#obtain [X,...X]
        # YY = np.vstack([Y for _ in range(n)]).T.reshape([n*m,Y.shape[0]]).T#reshape([Y.shape[0],n*m])
        # D = np.sqrt(sum(XX-YY)**2).T.reshape(m,n).T
        # minD = np.array([D[i,j] if D[i,j]>c else c for i in range(D.shape[0]) for j in range(D.shape[1])]).reshape(D.shape)
        # D = minD**p

        D = np.zeros([n,m])
        for j in range(m):
            #np.hstack([np.array([Y[:,j]],np.newaxis).T for _ in range(n)])
            D[:,j] = np.sqrt((sum((np.hstack([np.array([Y[:,j]],np.newaxis).T for _ in range(n)]))-X)**2).T)
        minD = np.array([D[i,j] if D[i,j]<c else c for i in range(D.shape[0]) for j in range(D.shape[1])]).reshape(D.shape)
        D = minD**p

        # Compute the cost with optimal assignment using Hungarian Algrithm
        row_ind,col_ind=linear_sum_assignment(D)
        cost = D[row_ind,col_ind].sum()
        # calculate the Final Distance
        dist = (1/max(m,n)*(c**p*np.abs(m-n)+cost))**(1/p)
        return dist

    def solver(self):
        return self._OPSA_fun(self.X,self.Y,self.c,self.p)
if __name__ == '__main__':
    X = {0: np.array([ 10.79523, -0.76643553,  0.37045356,  0.00437738]), 1: np.array([ 1.71480081, -0.79268797,  0.34578638, -0.02240102]), 2: np.array([ 1.6338837 , -0.75252312,  0.32811741, -0.04450454])}
    Y = {1: np.array([ 1.59461305, -0.64429735,  0.30792371,  0.03965508]), 2: np.array([ 1.68659741, -0.8577948 ,  0.37479998, -0.03048327]), 3: np.array([ 1.76226441, -0.88336732,  0.3207995 , -0.03625363])}
    score = OSPA(X,Y,1,2)
    print(score.X)
    print(score.Y)
    print(score.solver())
