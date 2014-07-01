# NCA

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import neighbors

class ONCA():
    
    def __init__(self,D,d,olamda=0.,maxiter=20,batchsize=20,monitor=True):
        self.D = D
        self.d = d
        self.lamda = 0.
        self.olamda = olamda
        self.Ae = 0.01*np.random.randn(D,d)
        self.Ap = 0.01*np.random.randn(D,d)
        self.eps = np.finfo(np.float).eps
        self.maxiter = maxiter        
        self.batchsize = batchsize
        self.monitor = monitor
        self.lr = 0.0005
        
    def Etransform(self,x):                
        return np.dot(x,self.Ae)

    def Ptransform(self,x):                
        return np.dot(x,self.Ap)

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

    def f_df(self,Ae,Ap,x,ye,yp):                     
        Ax = self.project(x,Ae)
        Pij,Pi = self.calc_softmax(Ax,ye)
        f = np.sum(Pi)        
        
        N = len(Pij)
        df1 = np.zeros(np.shape(Ae))
        df2 = np.zeros(np.shape(Ae))
        for i in range(N):            
            xik = x[i,:].reshape(1,-1) - x
            p = Pij[i,:].reshape(-1,1)
            tmp = Pi[i]*np.dot((p*xik).T,xik)
            df1 += np.dot(tmp,Ae)

            l = ye[i]
            dx = np.where(ye==l)[0]            
            xij = np.take(xik,dx,axis=0)
            p = np.take(Pij[i,:],dx).reshape(-1,1)
            tmp = np.dot((p*xij).T,xij)
            df2 += np.dot(tmp,Ae)
            
        dAe = (df1 - df2)/len(x)
        tmp1 = np.dot(np.dot(Ae,Ap.T),Ap)
        dAe = dAe - self.lamda*Ae/(self.D*self.d) - self.olamda*tmp1

        Ax = self.project(x,Ap)
        Pij,Pi = self.calc_softmax(Ax,yp)
        f += np.sum(Pi)        
        
        N = len(Pij)
        df1 = np.zeros(np.shape(Ap))
        df2 = np.zeros(np.shape(Ap))
        for i in range(N):            
            xik = x[i,:].reshape(1,-1) - x
            p = Pij[i,:].reshape(-1,1)
            tmp = Pi[i]*np.dot((p*xik).T,xik)
            df1 += np.dot(tmp,Ap)

            l = yp[i]
            dx = np.where(yp==l)[0]            
            xij = np.take(xik,dx,axis=0)
            p = np.take(Pij[i,:],dx).reshape(-1,1)
            tmp = np.dot((p*xij).T,xij)
            df2 += np.dot(tmp,Ap)

        dAp = (df1 - df2)/len(x)
        tmp1 = np.dot(np.dot(Ap,Ae.T),Ae)
        dAp = dAp - self.lamda*Ap/(self.D*self.d) - self.olamda*tmp1

        f -= self.lamda*(np.sum(Ap)**2)/(self.D*self.d) 
        f -= self.lamda*(np.sum(Ae)**2)/(self.D*self.d)
        f -= self.olamda*np.trace(np.dot(np.dot(Ae,Ae.T),np.dot(Ap,Ap.T)))
        return -f,-dAe,-dAp

    def evaluate(self,xtrn,yetrn,yptrn,xtst,yetst,yptst):
        Aetrn = self.Etransform(xtrn)
        Aetst = self.Etransform(xtst)
        Aptrn = self.Ptransform(xtrn)
        Aptst = self.Ptransform(xtst)

        clf = neighbors.KNeighborsClassifier(1, weights='uniform')
        clf.fit(Aetrn,yetrn)
        pred = clf.predict(Aetst)
        cm = np.zeros((2,2))
        for p,t in zip(pred,yetst):
            cm[t,p] += 1
        euwr = np.sum(np.diag(cm)/np.sum(cm,axis=1))/2

        clf = neighbors.KNeighborsClassifier(1, weights='uniform')
        clf.fit(Aptrn,yptrn)
        pred = clf.predict(Aptst)
        cm = np.zeros((2,2))
        for p,t in zip(pred,yptst):
            cm[t,p] += 1
        puwr = np.sum(np.diag(cm)/np.sum(cm,axis=1))/2
        
        return euwr,puwr
        
    def fit(self,x,ye,yp):
        L = len(x)
        if self.monitor == True:
            dx = np.arange(L)
            np.random.shuffle(dx)
            numtrn = np.round(0.7*L)
            dx1 = dx[0:numtrn]
            dx2 = dx[numtrn:]
            xtrn = np.take(x,dx1,axis=0)
            xtst = np.take(x,dx2,axis=0)
            yetrn = np.take(ye,dx1)
            yetst = np.take(ye,dx2)
            yptrn = np.take(yp,dx1)
            yptst = np.take(yp,dx2)
            
        numbatches = np.ceil(L/self.batchsize)
        i = 0
        costs = []
        while i < self.maxiter:
            cost = 0
            inds = np.arange(L)
            np.random.shuffle(inds)
            ix = np.take(x,inds,axis=0)
            iye = np.take(ye,inds)
            iyp = np.take(yp,inds)
            for j in range(0,L,self.batchsize):                
                bx = ix[j:(j+self.batchsize),:]
                bye = iye[j:(j+self.batchsize)]
                byp = iyp[j:(j+self.batchsize)]
                args = (bx,bye,byp)                

                f,dAe,dAp = self.f_df(self.Ae,self.Ap,*args)
                cost += f
                self.Ae -= self.lr*dAe
                self.Ap -= self.lr*dAp
            costs.append(cost/numbatches)            
            if self.monitor == True:
                eres,pres = self.evaluate(xtrn,yetrn,yptrn,xtst,yetst,yptst)
                print 'Iteration ' + str(i) + ' : ' + str(eres) + ',' + str(pres)
            else:
                print 'Iteration ' + str(i) + ' : ' + str(cost/numbatches)
            i += 1
            self.lr *= 0.95
        self.C = costs
    
    def get_costs(self):
        return self.C