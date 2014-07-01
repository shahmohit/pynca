import numpy as np
import os

# Generic
def read_data(path,ftype='npy'):
    if ftype == 'npy':
        data = np.load(path)
    elif ftype == 'csv':
        data = np.loadtxt(path,delimiter=',')
    labs = data[:,0]
    feats = data[:,1:]
    return feats,labs

# EMO-DB Single Speaker
def read_emodb_sd(path,ftype='npy'):
    if ftype == 'npy':
        data = np.load(path)
    elif ftype == 'csv':
        data = np.loadtxt(path,delimiter=',')
    plabs = data[:,-1]
    cvlabs = data[:,-2]
    elabs = data[:,-3]
    labs = data[:,-4]
    feats = data[:,0:-4]
    return feats,plabs,cvlabs,elabs,labs

# EMO-DB Speaker Independent
def read_emodb_si(spkr,task,dims):    
    ddir = '../data'
    spkrs = range(10)
    spkrs.remove(spkr)

    trn = np.zeros((1,dims))
    etrnlabs = []
    ctrnlabs = []
    ptrnlabs = []

    tst = np.zeros((1,dims))
    etstlabs = []
    ctstlabs = []
    ptstlabs = []

    for sp in spkrs:
        path = os.path.join(ddir,str(sp)+'.'+task+'.txt')
        data = np.loadtxt(path,delimiter=',')
        ptrnlabs += list(data[:,-1])
        ctrnlabs += list(data[:,-2])
        etrnlabs += list(data[:,-3])
        trn = np.vstack((trn,data[:,0:-4]))
    
    path = os.path.join(ddir,str(spkr)+'.'+task+'.txt')
    data = np.loadtxt(path,delimiter=',')
    ptstlabs += list(data[:,-1])
    ctstlabs += list(data[:,-2])
    etstlabs += list(data[:,-3])
    tst = np.vstack((tst,data[:,0:-4]))
    
    data = {}
    data['trn'] = trn[1:,:]
    data['tst'] = tst[1:,:]
    data['etrnlabs'] = np.array(etrnlabs)
    data['etstlabs'] = np.array(etstlabs)
    data['ctrnlabs'] = np.array(ctrnlabs)-1
    data['ctstlabs'] = np.array(ctstlabs)-1
    data['ptrnlabs'] = np.array(ptrnlabs)
    data['ptstlabs'] = np.array(ptstlabs)
    return data

def read_emodb_si_ids(spkr,task):    
    ddir = '../data'
    spkrs = range(10)
    spkrs.remove(spkr)

    idtrn = []
    idtst = []

    for sp in spkrs:
        path = os.path.join(ddir,str(sp)+'.'+task+'.id.txt')
        data = np.loadtxt(path)
        idtrn += list(data)
    
    path = os.path.join(ddir,str(spkr)+'.'+task+'.id.txt')
    data = np.loadtxt(path)
    idtst += list(data)
    
    return np.array(idtrn), np.array(idtst)

# EMO-DB Speaker Independent, Phoneme Specific
def read_emodb(spkr,task,phoneme):
    ddir = '../data'
    symbols = []
    path = os.path.join(ddir,'symbols.txt')
    fi = open(path,'rb')
    for x in fi.readlines():
        symbols.append(x.rstrip('\n'))
    fi.close()
    symint = []
    for p in phoneme:
        symint.append(symbols.index(p))
    dims = 68
    newdata = np.zeros((1,dims))
    spkrs = range(10)
    S = []
    for s in spkrs:
        path = os.path.join(ddir,str(s)+'.'+task+'.txt')
        data = np.loadtxt(path,delimiter=',')
        plabs = data[:,-1]
        for sym in symint:
            dx = np.where(plabs==sym)[0]
            if len(dx) > 0:
                newdata = np.vstack((newdata,np.take(data,dx,axis=0)))
                S += [s for x in range(len(dx))]
    newdata = newdata[1:,:]
    S = np.array(S).reshape(-1,1)
    dx = np.where(S==spkr)[0]
    tst = np.take(newdata,dx,axis=0)
    trn = np.delete(newdata,dx,axis=0)
    return trn,tst
    