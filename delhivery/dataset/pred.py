from keras.models import load_model
import scipy.misc
import cv2
import pandas as pd
import operator
import numpy as np
import h5py
import matplotlib.pyplot as plt
root_dir="/home/delhivery"
rel_path="/Desktop/dataset"
test=pd.read_csv(root_dir+rel_path+"/Test.csv")
model= load_model('Digit_classifier.18-0.065.hdf5')
img_test=np.zeros((21000,28,28))
x=range(0,21000)

for i in range(len(test)):
    img=scipy.misc.imread(root_dir+rel_path+"/Train/Images/test/"+test.filename[i])
    img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
    img_test[i]=img

img_test = img_test.reshape(img_test.shape[0], 1,28, 28).astype('float32')
img_test=img_test / 255
preds = model.predict (img_test, verbose=1)
plt.figure()
plt.plot(x,preds)
plt.show()
index=[]
for ind in range(len(test)):
	index.append(max(enumerate(preds[ind]), key=operator.itemgetter(1))[0])
test['label']=index
test.to_csv('my6Submission.csv',index=False)
