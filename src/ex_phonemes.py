import utils
import nca
import numpy as np
from sklearn import neighbors,svm,preprocessing
from sklearn.decomposition import pca
import matplotlib.pyplot as plt

R = 1
pscore = 0
nscore = 0
phones = ['i']
task = 'vs'

trn,tst = utils.read_emodb(0,task,phones)
trndata = trn[:,0:65]
tstdata = tst[:,0:65]
trnlabs = trn[:,-2]
tstlabs = tst[:,-2]

model = pca.PCA(n_components=2)
model.fit(trndata)
Ptrndata = model.transform(trndata)
Ptstdata = model.transform(tstdata)

nn = 2        
#clf = svm.LinearSVC()
clf = neighbors.KNeighborsClassifier(nn, weights='uniform')
clf.fit(Ptrndata,trnlabs)
pred = clf.predict(Ptstdata)
pcm = np.zeros((2,2))
for p,t in zip(pred,tstlabs):
    pcm[t,p] += 1
uwr = np.sum(np.diag(pcm)/np.sum(pcm,axis=1))/2
print uwr
        
D = np.shape(trndata)[1]
model = nca.NCA(D,2,maxiter=50,batchsize=100)
model.fit(trndata,trnlabs)
Atrndata = model.transform(trndata)
Atstdata = model.transform(tstdata)
costs = model.get_costs()

#clf = svm.LinearSVC()
clf = neighbors.KNeighborsClassifier(nn, weights='uniform')
clf.fit(Atrndata,trnlabs)
pred = clf.predict(Atstdata)
ncm = np.zeros((2,2))
for p,t in zip(pred,tstlabs):
    ncm[t,p] += 1
uwr = np.sum(np.diag(ncm)/np.sum(ncm,axis=1))/2
print uwr

dx1 = np.where(tstlabs==0)[0]
dx2 = np.where(tstlabs==1)[0]

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
