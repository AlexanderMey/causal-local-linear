import numpy as np
#### Data creation of Lorenz systems that can couple and decouple via the couplings vector coup:

### The couplings vector. The first dim is the start end end of different coupling intervals.
### The second dim is the strength of the couplings vector, where 0 means no coupling. 
### Note that a too high coulingsvector will make the system diverge
coup=np.array([[0,10000,20000,30000,40000,60000],[0,0,0,2,2,2]])
###

T=int(coup[0,-1]) 
X=np.zeros((T+1,6))
X[0,:]=[1,0.97,0.99,1,0.97,0.99]

                    
for r in range(len(coup[0,:])-1):
    alpha=coup[1,r]
    for t in range(int(coup[0,r]),int(coup[0,r+1])):
        if t<5:
            x_new=X[0,:]
        else:
    
            x_new=[X[t,0]*0.9+0.1*X[t,1],\
                   0.25*X[t,0]-0.01*X[t,0]*X[t,2]+0.99*X[t,1]+alpha*0.001*X[t-5,4]**2,\
                   0.01*X[t,0]*X[t,1]+0.9733*X[t,2],\
                   X[t,3]*0.9+0.1*X[t,4],\
                   X[t,3]*0.28-X[t,3]*X[t,5]*0.01+0.99*X[t,4]+alpha*0.0005*X[t-3,1]**2,\
                   0.01*X[t,3]*X[t,4]+0.9733*X[t,5]]

        X[t+1,:]=x_new

np.savetxt('Lorenz.csv',X,delimiter=",")