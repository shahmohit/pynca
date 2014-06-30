import numpy as np

def read_data(path,ftype='npy'):
    if ftype == 'npy':
        data = np.load(path)
    elif ftype == 'csv':
        data = np.loadtxt(path,delimiter=',')
    labs = data[:,0]
    feats = data[:,1:]
    return feats,labs

def read_emotion_data(path,ftype='npy'):
    if ftype == 'npy':
        data = np.load(path)
    elif ftype == 'csv':
        data = np.loadtxt(path,delimiter=',')
    plabs = data[:,-1]
    elabs = data[:,-2]
    labs = data[:,-3]
    feats = data[:,0:-3]
    return feats,plabs,elabs,labs    