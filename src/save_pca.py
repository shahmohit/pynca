import utils
import numpy as np
from sklearn.decomposition import pca
import os

mdir = '../models'

spkrs = 10
P = 13
fdims = 65

task = 'as'
for spkr in range(spkrs):
    data = utils.read_emodb_si(spkr,task,fdims)
    trndata = data['trn']
    
    m = pca.PCA(n_components=P)
    m.fit(trndata)
    Pca = m.components_
    Pca = Pca.T
    
    path = os.path.join(mdir,'Pca.'+str(spkr)+'.'+task+'.npy')
    np.save(path,Pca)