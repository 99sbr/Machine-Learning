import libmr
import numpy as np
import pandas as pd
import operator
import scipy.spatial.distance as spd

# params and configuratoins
NCLASSES = 5
ALPHA_RANK = 5
WEIBULL_TAIL_SIZE = 20

# load features
activation_vector=np.load('intermediate.npy')
y_pred=np.load('y_pred.npy')
y_val=np.load('y_val.npy')
# get predicted class
preds=[]
for ind in range(len(y_pred)):
    preds.append(max(enumerate(y_pred[ind]), key=operator.itemgetter(1))[0])

# creating dataframes in padas
activation_frame=Dme(activation_vector)
pred=pd.DataFrame(preds)
true=pd.DataFrame(y_val)
pred.columns=['y_pred']
true.columns=['y_true']
df=pd.concat([pred,true,frame],axis=1)