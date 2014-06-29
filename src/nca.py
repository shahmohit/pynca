# NCA

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

class NCA():
    
    def __init__(self,D,d,lamda=0.):
        self.D = D
        self.d = d
        self.lamda = lamda
        self.A = 0.01*np.random.randn(D,d)
        self.eps = np.finfo(np.float).eps
        
    def transform(self,x):
        return np.dot(x,self.A)
    
    def calc_softmax(self,x):
        dist = np.exp(-(pairwise_distances(x,metric='l2')))
        np.fill_diagonal(dist,0)
        dist = dist/np.sum(dist,axis=1).reshape(-1,1)
        dist = np.maximum(dist,self.eps)
        return dist
    
    def f_df_cost(self,x,y,A,Pij):
        N = len(Pij)
        Pi = np.zeros(N)
        for i in range(N):
            l = y[i]
            dx = np.where(y==l)[0]
            Pi[i] = np.sum(np.take(Pij[i,:],dx))        
        return Pi
        
    def fit(self,x,y):
        Ax = self.transform(x)
        Pij = self.calc_softmax(Ax)
        Pi = self.f_df_cost(x,y,self.A,Pij)
        return Ax,Pij,Pi
