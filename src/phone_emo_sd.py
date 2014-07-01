import utils
import onca
import numpy as np
from sklearn import neighbors,svm,preprocessing
from sklearn.decomposition import pca
import matplotlib.pyplot as plt

R = 1
trn_ratio = 50.
pscore = 0
nscore = 0
loc = '../data/9.vs.txt'

data,plabs,cvlabs,elabs,labs = utils.read_emodb_sd(loc,'csv')
cvlabs = cvlabs-1
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
    etrnlabs = np.take(elabs,trndx)
    ctrnlabs = np.take(cvlabs,trndx)
    tstdata = np.take(data,tstdx,axis=0)
    etstlabs = np.take(elabs,tstdx)
    ctstlabs = np.take(cvlabs,tstdx)

    model = pca.PCA(n_components=2)
    model.fit(trndata)
    Ptrn = model.transform(trndata)
    Ptst = model.transform(tstdata)

    clf = neighbors.KNeighborsClassifier(10, weights='uniform')
    clf.fit(Ptrn,etrnlabs)
    pred = clf.predict(Ptst)
    cm = np.zeros((2,2))
    for p,t in zip(pred,etstlabs):
        cm[t,p] += 1
    peuwr = np.sum(np.diag(cm)/np.sum(cm,axis=1))/2

    clf = neighbors.KNeighborsClassifier(10, weights='uniform')
    clf.fit(Ptrn,ctrnlabs)
    pred = clf.predict(Ptst)
    cm = np.zeros((2,2))
    for p,t in zip(pred,ctstlabs):
        cm[t,p] += 1
    ppuwr = np.sum(np.diag(cm)/np.sum(cm,axis=1))/2
    
    print 'PCA: ' + str(peuwr) + ',' + str(ppuwr)
    
    D = np.shape(data)[1]
    model = onca.ONCA(D,2,olamda=5.0,maxiter=30,batchsize=100)
    model.fit(trndata,etrnlabs,ctrnlabs)
    
    Aetrn = model.Etransform(trndata)
    Aetst = model.Etransform(tstdata)

    Aptrn = model.Ptransform(trndata)
    Aptst = model.Ptransform(tstdata)

    costs = model.get_costs()

    clf = neighbors.KNeighborsClassifier(10, weights='uniform')
    clf.fit(Aetrn,etrnlabs)
    pred = clf.predict(Aetst)
    cm = np.zeros((2,2))
    for p,t in zip(pred,etstlabs):
        cm[t,p] += 1
    aeuwr = np.sum(np.diag(cm)/np.sum(cm,axis=1))/2

    clf = neighbors.KNeighborsClassifier(10, weights='uniform')
    clf.fit(Aptrn,ctrnlabs)
    pred = clf.predict(Aptst)
    cm = np.zeros((2,2))
    for p,t in zip(pred,ctstlabs):
        cm[t,p] += 1
    apuwr = np.sum(np.diag(cm)/np.sum(cm,axis=1))/2

    print 'NCA: ' + str(aeuwr) + ',' + str(apuwr)    
    

#print pscore/float(R)
#print nscore/float(R)
fig = plt.figure()

dx1 = np.where(etstlabs==0)[0]
dx2 = np.where(etstlabs==1)[0]

et1 = np.take(Aetst,dx1,axis=0)
et2 = np.take(Aetst,dx2,axis=0)

at1 = np.take(Aptst,dx1,axis=0)
at2 = np.take(Aptst,dx2,axis=0)

pt1 = np.take(Ptst,dx1,axis=0)
pt2 = np.take(Ptst,dx2,axis=0)

ax = fig.add_subplot(231)
ax.scatter(pt1[:,0],pt1[:,1],color='b')
ax.scatter(pt2[:,0],pt2[:,1],color='g')

ax = fig.add_subplot(232)
ax.scatter(et1[:,0],et1[:,1],color='b')
ax.scatter(et2[:,0],et2[:,1],color='g')

ax = fig.add_subplot(233)
ax.scatter(at1[:,0],at1[:,1],color='b')
ax.scatter(at2[:,0],at2[:,1],color='g')

dx1 = np.where(ctstlabs==0)[0]
dx2 = np.where(ctstlabs==1)[0]

et1 = np.take(Aetst,dx1,axis=0)
et2 = np.take(Aetst,dx2,axis=0)

at1 = np.take(Aptst,dx1,axis=0)
at2 = np.take(Aptst,dx2,axis=0)

pt1 = np.take(Ptst,dx1,axis=0)
pt2 = np.take(Ptst,dx2,axis=0)

ax = fig.add_subplot(234)
ax.scatter(pt1[:,0],pt1[:,1],color='b')
ax.scatter(pt2[:,0],pt2[:,1],color='g')

ax = fig.add_subplot(235)
ax.scatter(et1[:,0],et1[:,1],color='b')
ax.scatter(et2[:,0],et2[:,1],color='g')

ax = fig.add_subplot(236)
ax.scatter(at1[:,0],at1[:,1],color='b')
ax.scatter(at2[:,0],at2[:,1],color='g')
