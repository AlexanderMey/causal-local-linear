import numpy as np
import math
import random
import copy
import itertools
from heapq import nlargest


def main(x,y,B,l):
    E=len(x)
    n,d=x[0].shape
    temp=[list(itertools.combinations(range(d), k)) for k in range(0,d)]
    subsets = [item for sublist in temp for item in sublist]
    dic={}
    for ind in subsets:
        R=[]
        for i in range(E):
            n,d=x[i].shape
            if len(ind)==0:
                R.append(y[i]-np.mean(y[i]))
            else:
                xtemp=x[i][:,ind]
                # xtemp=np.concatenate((xtemp,np.ones((n,1))),axis=1)
                ind=np.array(ind)
                beta_hat=np.linalg.inv(xtemp.T@xtemp)@(xtemp.T)@y[i]
                R.append(y[i]-xtemp@beta_hat)
        dic[tuple(ind)]=copy.copy(R)
    S_ini=[]
    for k in dic:
        Res=dic[k]
        pval=0
        n=len(Res)
        for b in range(B):
            a=np.random.chisquare(n-len(k),E)
            T=[np.sum(r**2) for r in Res]
            T_b=np.sort(a,kind='mergesort')
            T_data=np.sort(T,kind='mergesort')
            if np.sum(T_data[:l])/np.sum(T_data[-l:])>np.sum(T_b[:l])/np.sum(T_b[-l:]):
                pval+=1/B
        if pval>0.05:
            S_ini.append(k)
    return [set(S_ini[k]) for k in range(len(S_ini))]


