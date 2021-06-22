# -*- coding=utf-8 -*-
### implement the Murty Algorithm by single thread (Standard)
import sys
import numpy as np
#from copy import copy, deepcopy
from scipy.optimize import linear_sum_assignment
###################
INF = 10**9
def Hungarian(matrix):
    """最大图匹配"""
    rows_idx, cols_idx = linear_sum_assignment(matrix)
    cost = matrix[rows_idx, cols_idx].sum()
    #assign = [[x,y] for x,y in zip(rows_idx,cols_idx)]
    return cols_idx, cost

def MurtyPartition(N, a, type):
    """
     MurtyPartition partitioin node N with its minimum assignment a
     input:
      N - in Murty's original paper, N is a "node", i.e. a non empty
      subset of A, which contains all assignment schemes.
      a - a nMeas*1 vector containing one assignment scheme.
      type - type == 0 for N-to-N assignment problem, type == 1 for
          M-to-N assignment problem, where M > N, e.g. assign M jobs to
          N worker.
    Output:
      nodeList - containing the list of partition of N. The
          union of all assignments to all partitions and assignment {a}
          forms a complete set of assignments to N.
    """
    a = np.array(a).reshape(-1,1)
    nMeas = len(a)
    tmp = np.arange(nMeas).reshape(-1,1) #index col
    a = np.hstack([tmp,a])
    aset = {tuple(elem) for elem in iter(a.tolist())}
    inset = {tuple(elem) for elem in iter(N[0])}
    a1set = inset.intersection(aset)
    a2set = aset.difference(a1set)
    a2 = sorted(list(a2set),key=lambda x:x[0])
    nodelist = []
    length = len(a2set)-1 if type==0 else len(a2set)
    for i in range(length):
        if i == 0:
            Inclu = N[0]
        else:
            tmp = np.array([list(x) for x in a2[:i]])
            if N[0].size == 0:
                Inclu= tmp
            else:
                Inclu = np.vstack([N[0],tmp])
        tmp1 = np.array([list(a2[i])])
        if N[1].size == 0:
            Exclu = tmp1
        else:
            Exclu = np.vstack([N[1],tmp1])
        res = [Inclu,Exclu]
        nodelist.append(res)
    return nodelist

def murty(costMat, k):
    """
    Murty's algorithm finds out the kth minimum assignments, k = 1, 2, ...
    Syntax:
      solution = murty(costMat, k)
    In:
       costMat - nMeas*nTarg cost matrix.
       k - the command number controlling the output size.

    Out:
       solution - array containing the minimum, 2nd minimum, ...,
           kth minimum assignments and their costs. Each solution{i}
           contains {assgmt, cost} where assgmt is an nMeas*1 matrix
           giving the ith minimum assignment; cost is the cost of this
           assignment.
    """
    solution = [[] for _ in range(k)]
    t = 0
    assgmt, cost = Hungarian(costMat)
    solution[0] = [assgmt,cost]
    nodeRec = [np.array([]), np.array([])]
    assgmtRec = assgmt
    nodeList = MurtyPartition(nodeRec,assgmtRec,1)
    while t<k-1:
        minCost = INF #float("Inf")
        idxRec = -1
        #try to find one node in the nodeList with the minimum cost
        #print("length",len(nodeList))
        for i in range(len(nodeList)):
            node = nodeList[i].copy()
            Inclu = node[0]
            Exclu = node[1]
            mat = costMat.copy()
            #print(len(Inclu))
            for j in range(len(Inclu)):
                best=mat[Inclu[j,0],Inclu[j,1]]
                mat[Inclu[j,0],:] = INF
                mat[Inclu[j,0],Inclu[j,1]]=best
            for j in range(len(Exclu)):
                mat[Exclu[j,0],Exclu[j,1]]=INF
            assgmt,cost = Hungarian(mat)
            #print(assgmt)
            # if -1 in assgmt:
            #     continue
            if cost < minCost:
                minCost = cost
                nodeRec = node
                assgmtRec = assgmt
                idxRec = i
        if idxRec == -1:
            for i in range(t,k):
                solution[i] = solution[t].copy()
            t = k
            #print("adadad")
        else:
            t += 1
            solution[t] = [assgmtRec, minCost]
            lenNodeSet = set(range(len(nodeList)))
            idxSet = {idxRec}
            idx = lenNodeSet.difference(idxSet)
            nodetmp = [nodeList[i] for i in idx]
            nodeList = nodetmp + MurtyPartition(nodeRec, assgmtRec, 1)
    return solution

if __name__ == '__main__':
    costMat = np.loadtxt("input.test")
    k = 20
    #print(murty(costMat,k))
    for assign, cost in murty(costMat,k):
        print(assign,cost)
