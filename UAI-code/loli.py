import numpy as np
import math
import random
import copy
import itertools
from heapq import nlargest


def gauss(x,y,B=100,l=1,alpha=0.1,lam=0,rs=False,intercept=True,pvalues=False):
    if rs>0:
        np.random.seed(rs)
    E=len(x)
    n,d=x[0].shape
    temp=[list(itertools.combinations(range(d), k)) for k in range(0,d+1)]
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
        Simulations={}
        for i in range(E):
            n,d=x[i].shape
            m=len(ind)+intercept
            if (n,0) not in Simulations:
                Simulations[(i,n,intercept+d)]=(np.random.chisquare(n-d-1,B))
                for q in range(1,intercept+d):
                    Simulations[(i,n,intercept+d-q)]=Simulations[(i,n,intercept+d-q+1)]+(np.random.normal(size=(B)))**2
            if len(ind)==0:
                Res[i]=np.sum((y[i]-np.mean(y[i]))**2)
            else:
                xtemp=copy.copy(x[i][:,ind])
                if intercept:
                    xtemp=np.concatenate((xtemp,np.ones((n,1))),axis=1)
 
                beta_hat=np.linalg.inv(xtemp.T@xtemp+lam*np.eye(m))@(xtemp.T)@y[i]
                Res[i]=np.sum((y[i]-xtemp@beta_hat)**2)

                T_mc[i,:]=Simulations[(i,n,m)] 
        if l==1:
            pval=1/B*np.sum(np.min(Res)*np.max(T_mc,axis=0)>np.max(Res)*np.min(T_mc,axis=0))
        else:
            T_b=np.sort(T_mc,kind='mergesort',axis=0)
            T_data=np.sort(Res,kind='mergesort')
            pval=1/B*np.sum(np.sum(T_data[:l])*np.sum(T_b[-l:,:],axis=0)>np.sum(T_b[:l,:],axis=0)*np.sum(T_data[-l:]))
        if pval>alpha:
            if pvalues:
                S_ini.append((ind,pval))
            else:
                S_ini.append(ind)
    if pvalues:
        return [S_ini[k] for k in range(len(S_ini))]
    return [set(S_ini[k]) for k in range(len(S_ini))]



def clt(x,y,B=100,l=1,alpha=0.1,lam=0,rs=False,intercept=True,pvalues=False):
    if rs>0:
        np.random.seed(rs)
    E=len(x)
    n,d=x[0].shape
    temp=[list(itertools.combinations(range(d), k)) for k in range(0,d+1)]
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
        Simulations={}
        for i in range(E):
            n,d=x[i].shape
            m=len(ind)+intercept
            if (n,0) not in Simulations:
                Simulations[(i,n,intercept+d)]=(np.random.normal(0,1,size=B))
                for q in range(1,intercept+d):
                    Simulations[(i,n,intercept+d-q)]=(np.random.normal(0,1,size=B))
            if len(ind)==0:
                Res[i]=np.sum((y[i]-np.mean(y[i]))**2)
            else:
                xtemp=copy.copy(x[i][:,ind])
                if intercept:
                    xtemp=np.concatenate((xtemp,np.ones((n,1))),axis=1)
 
                beta_hat=np.linalg.inv(xtemp.T@xtemp+lam*np.eye(m))@(xtemp.T)@y[i]
                Res[i]=np.sum((y[i]-xtemp@beta_hat)**2)-np.mean((y[i]-xtemp@beta_hat)**2)

                T_mc[i,:]=Simulations[(i,n,m)] 
        if l==1:
            # print(ind)
            T_mc_all=np.random.normal(0,(1/n*np.median(Res))**(1/2),size=(n,E,B))
            T_mc=np.sum(T_mc_all**2,axis=0)
            # print(np.min(Res))
            # print(np.min(T_mc,axis=0))
            # print(np.max(Res))
            # print(np.max(T_mc,axis=0))

            pval=1/B*np.sum(np.min(Res)*np.max(T_mc,axis=0)>np.max(Res)*np.min(T_mc,axis=0))
        else:
            T_b=np.sort(T_mc,kind='mergesort',axis=0)
            T_data=np.sort(Res,kind='mergesort')
            pval=1/B*np.sum(np.sum(T_data[:l])*np.sum(T_b[-l:,:],axis=0)>np.sum(T_b[:l,:],axis=0)*np.sum(T_data[-l:]))
        if pval>alpha:
            if pvalues:
                S_ini.append((ind,pval))
            else:
                S_ini.append(ind)
    if pvalues:
        return [S_ini[k] for k in range(len(S_ini))]
    return [set(S_ini[k]) for k in range(len(S_ini))]

import numpy as np
import math
import random
import copy
import itertools
from heapq import nlargest


def trunc(x,y,B=100,l=1,alpha=0.1,lam=0,trunc=5,rs=False,intercept=True,pvalues=False):
    if rs>0:
        np.random.seed(rs)
    E=len(x)
    n,d=x[0].shape
    n=n-trunc
    temp=[list(itertools.combinations(range(d), k)) for k in range(0,d+1)]
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
        Simulations={}
        for i in range(E):
            n,d=x[i].shape
            m=len(ind)+intercept
            if (n,0) not in Simulations:
                Simulations[(i,n,intercept+d)]=(np.random.chisquare(n-d-1,B))
                for q in range(1,intercept+d):
                    Simulations[(i,n,intercept+d-q)]=Simulations[(i,n,intercept+d-q+1)]+(np.random.normal(size=(B)))**2
            if len(ind)==0:
                Res[i]=np.sum((y[i]-np.mean(y[i]))**2)
            else:
                xtemp=copy.copy(x[i][:,ind])
                if intercept:
                    xtemp=np.concatenate((xtemp,np.ones((n,1))),axis=1)
 
                beta_hat=np.linalg.inv(xtemp.T@xtemp+lam*np.eye(m))@(xtemp.T)@y[i]
                residuals=(y[i]-xtemp@beta_hat)**2
                order=residuals.argsort()
                trunc_ind=order[trunc:-trunc]
                residuals=residuals[trunc_ind]
                print('check')
                Res[i]=np.sum(residuals)

                T_mc[i,:]=Simulations[(i,n,m)]
        if l==1:
            pval=1/B*np.sum(np.min(Res)*np.max(T_mc,axis=0)>np.max(Res)*np.min(T_mc,axis=0))
        else:
            T_b=np.sort(T_mc,kind='mergesort',axis=0)
            T_data=np.sort(Res,kind='mergesort')
            pval=1/B*np.sum(np.sum(T_data[:l])*np.sum(T_b[-l:,:],axis=0)>np.sum(T_b[:l,:],axis=0)*np.sum(T_data[-l:]))
        if pval>alpha:
            if pvalues:
                S_ini.append((ind,pval))
            else:
                S_ini.append(ind)
    if pvalues:
        return [S_ini[k] for k in range(len(S_ini))]
    return [set(S_ini[k]) for k in range(len(S_ini))]
