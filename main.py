from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
newdata=np.genfromtxt('abalone.data.txt',delimiter=',',dtype="|S5")
for i in range(0,4177):
    if newdata[i,0]=='M':
        newdata[i,0]=1
    elif newdata[i,0]=='F':
        newdata[i,0]=-1
    else:
        newdata[i,0]=0
y=newdata[:,-1]
x=newdata[:,1:9]
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
clf=linear_model.LinearRegression()
X_train= np.array(X_train, dtype = 'float_')
X_test= np.array(X_test, dtype = 'float_')
y_train= np.array(y_train, dtype = 'float_')
y_test= np.array(y_test, dtype = 'float_')
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

print('Coefficients: \n', clf.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((clf.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % clf.score(X_test, y_test))

