# NCA

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from minimize import minimize

class NCA():
    
    def __init__(self,D,d,lamda=0.,maxiter=20,batchsize=20):
        self.D = D
        self.d = d
        self.lamda = lamda
        self.A = 0.01*np.random.randn(D,d)
        self.eps = np.finfo(np.float).eps
        self.maxiter = maxiter        
        self.batchsize = batchsize
    def transform(self,x):        
        return np.dot(x,self.A)

    def project(self,x,A):        
        return np.dot(x,A)
    
    def calc_softmax(self,Ax,y):
        Pij = np.exp(-(pairwise_distances(Ax,metric='l2')))
        np.fill_diagonal(Pij,0)
        Pij = Pij/np.sum(Pij,axis=1).reshape(-1,1)
        Pij = np.maximum(Pij,self.eps)

        N = len(Pij)
        Pi = np.zeros(N)
        for i in range(N):
            l = y[i]
            dx = np.where(y==l)[0]            
            Pi[i] = np.sum(np.take(Pij[i,:],dx))        
        return Pij, Pi

    def f_df(self,A,x,y):                
        Ax = self.project(x,A)
        Pij,Pi = self.calc_softmax(Ax,y)
        f = np.sum(Pi)
        f = f - self.lamda*(np.sum(A)**2)/(self.D*self.d);
        
        N = len(Pij)
        df1 = np.zeros(np.shape(A))
        df2 = np.zeros(np.shape(A))
        for i in range(N):            
            xik = x[i,:].reshape(1,-1) - x
            p = Pij[i,:].reshape(-1,1)
            tmp = Pi[i]*np.dot((p*xik).T,xik)
            df1 += np.dot(tmp,A)

            l = y[i]
            dx = np.where(y==l)[0]            
            xij = np.take(xik,dx,axis=0)
            p = np.take(Pij[i,:],dx).reshape(-1,1)
            tmp = np.dot((p*xij).T,xij)
            df2 += np.dot(tmp,A)
        df = (df1 - df2)
        df = df - self.lamda*A/(self.D*self.d);
        return -f,-df
                
    def fit(self,x,y):
        numbatches = np.ceil(len(x)/self.batchsize)
        i = 0
        costs = []
        while i < self.maxiter:
            cost = 0
            inds = np.arange(len(x))
            np.random.shuffle(inds)
            x = np.take(x,inds,axis=0)
            y = np.take(y,inds)
            for j in range(0,len(x),self.batchsize):
                bx = x[j:(j+self.batchsize),:]
                by = y[j:(j+self.batchsize)]
                args = (bx,by)
                A,C,_ = minimize(self.A,self.f_df,args,5)
                cost += C[0]
                self.A = np.array(A)
            costs.append(cost/numbatches)
            i += 1
        self.C = costs
    
    def get_costs(self):
        return self.C