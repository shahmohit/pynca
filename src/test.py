import utils
import nca
import numpy as np

loc = '/home/mohit/Documents/nca/wine.data'

data,labs = utils.read_data(loc,'csv')
#data = data - np.mean(data,axis=0)
D = np.shape(data)[1]
model = nca.NCA(D,5)
xfrm,sim,Pi = model.fit(data,labs)