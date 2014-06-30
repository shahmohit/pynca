import utils
import nca
import numpy as np
from sklearn import neighbors,svm,preprocessing
from sklearn.decomposition import pca
import matplotlib.pyplot as plt

R = 1
trn_ratio = 50.
pscore = 0
nscore = 0
loc = '../data/1.vs.txt'

data,plabs,elabs,labs = utils.read_emotion_data(loc,'csv')
#data = data - np.mean(data,axis=0)
#data = preprocessing.normalize(data)
ln = len(data)
numtrn = np.round(ln*trn_ratio/100)
for r in range(R):    
    inds = np.arange(len(data))
    np.random.shuffle(inds)
    trndx = inds[0:numtrn]
    tstdx = inds[numtrn:]
    trndata = np.take(data,trndx,axis=0)
    trnlabs = np.take(labs,trndx)
    etrnlabs = np.take(elabs,trndx)
    tstdata = np.take(data,tstdx,axis=0)
    tstlabs = np.take(labs,tstdx)
    etstlabs = np.take(elabs,tstdx)
    
    D = np.shape(data)[1]
    model = nca.NCA(D,20,maxiter=30,batchsize=100)
    model.fit(trndata,trnlabs)
    Atrndata = model.transform(trndata)
    Atstdata = model.transform(tstdata)
    costs = model.get_costs()
    
    model = pca.PCA(n_components=20)
    model.fit(trndata)
    Ptrndata = model.transform(trndata)
    Ptstdata = model.transform(tstdata)
            
    clf = svm.LinearSVC()
    clf.fit(Ptrndata,etrnlabs)
    pscore += clf.score(Ptstdata,etstlabs)
    
    clf = svm.LinearSVC()
    clf.fit(Atrndata,etrnlabs)
    nscore += clf.score(Atstdata,etstlabs)

print pscore/float(R)
print nscore/float(R)
'''
dx1 = np.where(etstlabs==0)[0]
dx2 = np.where(etstlabs==1)[0]

nt1 = np.take(Atstdata,dx1,axis=0)
nt2 = np.take(Atstdata,dx2,axis=0)

pt1 = np.take(Ptstdata,dx1,axis=0)
pt2 = np.take(Ptstdata,dx2,axis=0)

fig = plt.figure()
ax = fig.add_subplot(121)
ax.scatter(pt1[:,0],pt1[:,1],color='b')
ax.scatter(pt2[:,0],pt2[:,1],color='g')

ax = fig.add_subplot(122)
ax.scatter(nt1[:,0],nt1[:,1],color='b')
ax.scatter(nt2[:,0],nt2[:,1],color='g')
'''