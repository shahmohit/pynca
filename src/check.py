import utils
import numpy as np

spkr = 0
P = 13
fdims = 65
task = 'vs'

data = utils.read_emodb_si(spkr,task,fdims)
trndata = data['trn']
etrnlabs = data['etrnlabs']
ctrnlabs = data['ctrnlabs']
tstdata = data['tst']
etstlabs = data['etstlabs']
ctstlabs = data['ctstlabs']

Ae = np.load('../models/Ae.0.vs.npy')
tst = np.dot(tstdata,Ae)
tst1 = np.load('../aetst1.npy')