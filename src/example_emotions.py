import utils
import nca
import numpy as np
from sklearn import neighbors,svm
from sklearn.decomposition import pca
import matplotlib.pyplot as plt

R = 1
trn_ratio = 50.
pscore = 0
nscore = 0
loc = '../data/8.vs.txt'

data,labs = utils.read_data(loc,'csv')
#data = data[5000:,:]
#labs = labs[5000:]+1
#data = data - np.mean(data,axis=0)
ln = len(data)
numtrn = np.round(ln*trn_ratio/100)
for r in range(R):    
    inds = np.arange(len(data))
    np.random.shuffle(inds)
    trndx = inds[0:numtrn]
    tstdx = inds[numtrn:]
    trndata = np.take(data,trndx,axis=0)
    trnlabs = np.take(labs,trndx)
    tstdata = np.take(data,tstdx,axis=0)
    tstlabs = np.take(labs,tstdx)
    
    D = np.shape(data)[1]
    model = nca.NCA(D,2,maxiter=20,lamda=0.002,batchsize=100)
    model.fit(trndata,trnlabs)
    Atrndata = model.transform(trndata)
    Atstdata = model.transform(tstdata)
    costs = model.get_costs()
    
    model = pca.PCA(n_components=2)
    model.fit(trndata)
    Ptrndata = model.transform(trndata)
    Ptstdata = model.transform(tstdata)
            
    clf = svm.LinearSVC()
    clf.fit(Ptrndata,trnlabs)
    pscore += clf.score(Ptstdata,tstlabs)
    
    clf = svm.LinearSVC()
    clf.fit(Atrndata,trnlabs)
    nscore += clf.score(Atstdata,tstlabs)

print pscore/float(R)
print nscore/float(R)
