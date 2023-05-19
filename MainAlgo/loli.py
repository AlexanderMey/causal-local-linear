import numpy as np
import math
import random
import copy
import itertools
from heapq import nlargest


def gauss(x,y,B,l):
    E=len(x)
    alpha=0.1
    n,d=x[0].shape
    temp=[list(itertools.combinations(range(d), k)) for k in range(0,d)]
    subsets = [item for sublist in temp for item in sublist]
    dic={}
    S_ini=[]
    suppsize=d
    for ind in subsets:
        ## In case we want early stopping for more power:
        # if len(ind)>suppsize:
        #     return [set(S_ini[k]) for k in range(len(S_ini))]
        ind=np.array(ind)
        Res=np.zeros(E)
        T_mc=np.zeros((E,B))
        for i in range(E):
            n,d=x[i].shape
            if len(ind)==0:
                Res[i]=np.sum((y[i]-np.mean(y[i]))**2)
            else:
                xtemp=copy.copy(x[i][:,ind])
                xtemp=np.concatenate((xtemp,np.ones((n,1))),axis=1)
                beta_hat=np.linalg.inv(xtemp.T@xtemp)@(xtemp.T)@y[i]
                Res[i]=np.sum((y[i]-xtemp@beta_hat)**2)

                T_mc[i,:]=(np.random.chisquare(n-len(ind)-1,B))
        T_b=np.sort(T_mc,kind='mergesort',axis=0)
        T_data=np.sort(Res,kind='mergesort')
        pval=1/B*np.sum(np.sum(T_data[:l])*np.sum(T_b[-l:,:],axis=0)>np.sum(T_b[:l,:],axis=0)*np.sum(T_data[-l:]))
        if pval>0.1:
            suppsize=len(ind)
            S_ini.append(ind)
    return [set(S_ini[k]) for k in range(len(S_ini))]


def shuffle(x,B):
    E=len(x)
    x_shuffle=[]
    for e in range(E):
        n,d=x[e].shape
        x_temp=np.zeros((B,n,d))
        for b in range(B):
            for i in range(d):
                c=(x[e][:,i].T).flatten()
                np.random.shuffle(c)
                x_temp[b,:,i]=c
        x_shuffle.append(x_temp)
        
    return x_shuffle


def noassumption(x,y,B,l):
    S1=[0,1,2,3,4]
    E=len(x)
    lam=0
    n,d=x[0].shape
    temp=[list(itertools.combinations(range(d), k)) for k in range(0,d)]
    subsets = [item for sublist in temp for item in sublist]
    dic={}
    S_ini=[]
    x_shuffle=shuffle(x,B)
    for ind in subsets:
        Res=np.zeros((E,B))
        T_mc=np.zeros((E,B))
        for i in range(E):
            n,d=x[i].shape
            if len(ind)==0:
#                 Res[i]=np.sum((y[i]-np.mean(y[i]))**2)
                Res[i]=np.sum((y[i])**2)
            else:
                comp=list(set(S1)-set(ind))
                xtemp=copy.copy(x[i])
                for b in range(B):
                    xtemp[:,comp]=(x_shuffle[i][b,:,comp]).T
                    beta_hat=np.linalg.inv(xtemp.T@xtemp+lam*np.eye(d))@(xtemp.T)@y[i]
                    Res[i,b]=np.sum((y[i]-xtemp@beta_hat)**2)
#         T_data=np.var(Res,axis=0)
        T_data=100*np.min(Res,axis=0)/np.max(Res,axis=0)
        print(ind)
        print(np.median(T_data),np.mean(T_data),np.std(T_data))
#         if something:
#             S_ini.append(ind)
    return [set(S_ini[k]) for k in range(len(S_ini))]


