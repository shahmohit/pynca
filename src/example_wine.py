import utils
import nca
import numpy as np
from sklearn import neighbors,preprocessing
from sklearn.decomposition import pca
import matplotlib.pyplot as plt

R = 100
trn_ratio = 50.
pscore = 0
nscore = 0
loc = '../data/wine.data'

data,labs = utils.read_data(loc,'csv')
data = data - np.mean(data,axis=0)
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
    model = nca.NCA(D,2,maxiter=10,lamda=0.002,batchsize=50)
    model.fit(trndata,trnlabs)
    Atrndata = model.transform(trndata)
    Atstdata = model.transform(tstdata)
    
    model = pca.PCA(n_components=2)
    model.fit(trndata)
    Ptrndata = model.transform(trndata)
    Ptstdata = model.transform(tstdata)
    
    n_neighbors = 2
    weights = 'uniform'
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(Ptrndata,trnlabs)
    pscore += clf.score(Ptstdata,tstlabs)
    
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(Atrndata,trnlabs)
    nscore += clf.score(Atstdata,tstlabs)

print pscore/float(R)
print nscore/float(R)

dx1 = np.where(tstlabs==1)[0]
dx2 = np.where(tstlabs==2)[0]
dx3 = np.where(tstlabs==3)[0]

nt1 = np.take(Atstdata,dx1,axis=0)
nt2 = np.take(Atstdata,dx2,axis=0)
nt3 = np.take(Atstdata,dx3,axis=0)

pt1 = np.take(Ptstdata,dx1,axis=0)
pt2 = np.take(Ptstdata,dx2,axis=0)
pt3 = np.take(Ptstdata,dx3,axis=0)

fig = plt.figure()
ax = fig.add_subplot(121)
ax.scatter(pt1[:,0],pt1[:,1],color='b')
ax.scatter(pt2[:,0],pt2[:,1],color='g')
ax.scatter(pt3[:,0],pt3[:,1],color='r')

ax = fig.add_subplot(122)
ax.scatter(nt1[:,0],nt1[:,1],color='b')
ax.scatter(nt2[:,0],nt2[:,1],color='g')
ax.scatter(nt3[:,0],nt3[:,1],color='r')
