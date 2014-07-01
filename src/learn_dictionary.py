import utils
import numpy as np
from sklearn.decomposition import pca
import scipy.cluster.vq as vq
import os

mdir = '../models'
ddir = '../dicts'
if not os.path.exists(ddir):
    os.mkdir(ddir)
    
spkr = 9
P = 13
fdims = 65
cbsize = 64

task = 'vs'
data = utils.read_emodb_si(spkr,task,fdims)
trndata = data['trn']

path = os.path.join(mdir,'Ae.'+str(spkr)+'.'+task+'.npy')
Ae = np.load(path)
aetrn = np.dot(trndata,Ae)
print 'Learning dictionary using Ae...'
cbk,_ = vq.kmeans(aetrn,cbsize,3)
path = os.path.join(ddir,'Ae.'+str(spkr)+'.'+task+'.'+str(cbsize)+'.npy')
np.save(path,cbk)
print 'Completed.'

path = os.path.join(mdir,'Ap.'+str(spkr)+'.'+task+'.npy')
Ap = np.load(path)
aptrn = np.dot(trndata,Ap)
print 'Learning dictionary using Ap...'
cbk,_ = vq.kmeans(aptrn,cbsize,3)
path = os.path.join(ddir,'Ap.'+str(spkr)+'.'+task+'.'+str(cbsize)+'.npy')
np.save(path,cbk)
print 'Completed.'

m = pca.PCA(n_components=P)
pcatrn = m.fit_transform(trndata)
print 'Learning dictionary using PCA...'
cbk,_ = vq.kmeans(pcatrn,cbsize,3)
path = os.path.join(ddir,'Pca.'+str(spkr)+'.'+task+'.'+str(cbsize)+'.npy')
np.save(path,cbk)
print 'Completed.'
