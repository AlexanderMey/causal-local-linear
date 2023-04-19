import numpy as np
import math
import random
import copy
import itertools
from heapq import nlargest


def main(x,y,B,l):
    E,n,d=x.shape
    temp=[list(itertools.combinations(range(d), k)) for k in range(0,d)]
    subsets = [item for sublist in temp for item in sublist]
    R=np.zeros((E,n))
    dic={}
    for ind in subsets:
        for i in range(E):
            if len(ind)==0:
                R[i,:]=y[i,:]
            else:
                ind=np.array(ind)
                beta_hat=np.linalg.inv(x[i,:,ind]@x[i,:,ind].T)@(x[i,:,ind])@y[i,:]
                R[i,:]=y[i,:]-x[i,:,ind].T@beta_hat
        dic[tuple(ind)]=copy.copy(R)
    S_ini=[]
    for k in dic:
        Res=dic[k]
        pval=0
        for b in range(B):
            a=np.random.chisquare(n-len(k),E)
            T=np.sum(Res**2,axis=1)
    #                 if np.min(T)/np.max(T)>np.min(a)/np.max(a):
    #                     pval+=1/B
            T_b=np.sort(a,kind='mergesort')
            T_data=np.sort(T,kind='mergesort')
            if np.sum(T_data[:l])/np.sum(T_data[-l:])>np.sum(T_b[:l])/np.sum(T_b[-l:]):
                pval+=1/B
        if pval>0.05:
            S_ini.append(k)
    return [set(S_ini[k]) for k in range(len(S_ini))]


