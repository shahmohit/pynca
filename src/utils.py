import numpy as np

def read_data(path,ftype='npy'):
    if ftype == 'npy':
        data = np.load(path)
    elif ftype == 'csv':
        data = np.loadtxt(path,delimiter=',')
    labs = data[:,0]
    feats = data[:,1:]
    return feats,labs