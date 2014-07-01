import utils
import numpy as np
from sklearn.decomposition import pca
import scipy.cluster.vq as vq
import os
from sklearn import preprocessing,svm

mdir = '../models'
ddir = '../dicts'

spkr = 9
P = 13
fdims = 65
cbsize = 64

doPCA = True

task = 'vs'
data = utils.read_emodb_si(spkr,task,fdims)
idtrn,idtst = utils.read_emodb_si_ids(spkr,task)
ltrn = data['etrnlabs']
ltst = data['etstlabs']
trndata = data['trn']
tstdata = data['tst']

path = os.path.join(mdir,'Ae.'+str(spkr)+'.'+task+'.npy')
Ae = np.load(path)
aetrn = np.dot(trndata,Ae)
aetst = np.dot(tstdata,Ae)
path = os.path.join(ddir,'Ae.'+str(spkr)+'.'+task+'.'+str(cbsize)+'.npy')
aecbk = np.load(path)
aewtrn,_ = vq.vq(aetrn,aecbk)
aewtst,_ = vq.vq(aetst,aecbk)

path = os.path.join(mdir,'Ap.'+str(spkr)+'.'+task+'.npy')
Ap = np.load(path)
aptrn = np.dot(trndata,Ap)
aptst = np.dot(tstdata,Ap)
path = os.path.join(ddir,'Ap.'+str(spkr)+'.'+task+'.'+str(cbsize)+'.npy')
apcbk = np.load(path)
apwtrn,_ = vq.vq(aptrn,aecbk)
apwtst,_ = vq.vq(aptst,aecbk)

if doPCA:
    path = os.path.join(mdir,'Pca.'+str(spkr)+'.'+task+'.npy')
    Pca = np.load(path)
    pcatrn = np.dot(trndata,Pca)
    pcatst = np.dot(tstdata,Pca)
    path = os.path.join(ddir,'Pca.'+str(spkr)+'.'+task+'.'+str(cbsize)+'.npy')
    pcacbk = np.load(path)
    pcawtrn,_ = vq.vq(pcatrn,aecbk)
    pcawtst,_ = vq.vq(pcatst,aecbk)

trnlabs = []
tstlabs = []
trnutts = list(set(idtrn))
tstutts = list(set(idtst))
ntrn = len(trnutts)
ntst = len(tstutts)
aetrn = np.zeros((ntrn,cbsize))
aetst = np.zeros((ntst,cbsize))
aptrn = np.zeros((ntrn,cbsize))
aptst = np.zeros((ntst,cbsize))

if doPCA:
    pcatrn = np.zeros((ntrn,cbsize))
    pcatst = np.zeros((ntst,cbsize))

k = 0
for idx in trnutts:
    dx = np.where(idtrn==idx)[0]
    l = np.take(ltrn,dx)
    trnlabs.append((np.sum(l)/float(len(l))).astype('int'))
    aew = np.take(aewtrn,dx)
    for w1 in aew:
        aetrn[k,w1] += 1
    apw = np.take(apwtrn,dx)
    for w1 in apw:
        aptrn[k,w1] += 1    
    if doPCA:
        pcaw = np.take(pcawtrn,dx)
        for w1 in pcaw:
            pcatrn[k,w1] += 1    
    k += 1

k = 0
for idx in tstutts:
    dx = np.where(idtst==idx)[0]
    l = np.take(ltst,dx)
    tstlabs.append((np.sum(l)/float(len(l))).astype('int'))
    aew = np.take(aewtst,dx)
    for w1 in aew:
        aetst[k,w1] += 1
    apw = np.take(apwtst,dx)
    for w1 in apw:
        aptst[k,w1] += 1    
    if doPCA:
        pcaw = np.take(pcawtst,dx)
        for w1 in pcaw:
            pcatst[k,w1] += 1
    k += 1

aetrn = preprocessing.normalize(aetrn)
aetst = preprocessing.normalize(aetst)

aptrn = preprocessing.normalize(aptrn)
aptst = preprocessing.normalize(aptst)

if doPCA:
    pcatrn = preprocessing.normalize(pcatrn)
    pcatst = preprocessing.normalize(pcatst)
    #pass

clf = svm.LinearSVC()
clf.fit(aetrn,trnlabs)
aepred = clf.predict(aetst)
cmae = np.zeros((2,2))
for p,t in zip(aepred,tstlabs):
    cmae[t,p] += 1
aewr = np.trace(cmae)/np.sum(cmae)
aeuwr = np.sum(np.diag(cmae)/np.sum(cmae,axis=1))/2
print aewr,aeuwr

clf = svm.LinearSVC()
clf.fit(aptrn,trnlabs)
appred = clf.predict(aptst)
cmap = np.zeros((2,2))
for p,t in zip(appred,tstlabs):
    cmap[t,p] += 1
apwr = np.trace(cmap)/np.sum(cmap)
apuwr = np.sum(np.diag(cmap)/np.sum(cmap,axis=1))/2
print apwr,apuwr

if doPCA:
    clf = svm.LinearSVC()
    clf.fit(pcatrn,trnlabs)
    pcapred = clf.predict(pcatst)
    cmpca = np.zeros((2,2))
    for p,t in zip(pcapred,tstlabs):
        cmpca[t,p] += 1

    pcawr = np.trace(cmpca)/np.sum(cmpca)
    pcauwr = np.sum(np.diag(cmpca)/np.sum(cmpca,axis=1))/2
    print pcawr,pcauwr