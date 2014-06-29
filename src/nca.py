# NCA

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from scipy.optimize import minimize

class NCA():
    
    def __init__(self,D,d,lamda=0.,maxiter=20,method='Newton-CG'):
        self.D = D
        self.d = d
        self.lamda = lamda
        self.A = np.random.randn(D*d)
        self.eps = np.finfo(np.float).eps
        self.maxiter = maxiter
        self.method = method
        
    def transform(self,x):
        A = self.A.reshape(self.D,self.d)
        return np.dot(x,A)
    
    def calc_softmax(self,x,y):
        Pij = np.exp(-(pairwise_distances(x,metric='l2')))
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
        A = A.reshape(self.D,self.d)
        Ax = self.transform(x)
        Pij,Pi = self.calc_softmax(Ax,y)
        f = -np.sum(Pi)

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
        df = (df1 - df2).ravel()
        return f,df
                
    def fit(self,x,y):
        options = {}
        options['maxiter'] = self.maxiter
        args = (x,y)        
        ans = minimize(self.f_df,self.A,args,method=self.method,jac=True,options=options)
        self.A = ans['x']        
