# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
df=pd.read_csv("ex1data1.txt",delimiter=',',names=['population','profit'])
frame=pd.DataFrame(data=df)

x=frame.drop(['profit'],axis=1)
y=frame.drop(['population'],axis=1)
clf=linear_model.LinearRegression()
X_train, X_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2, random_state=40)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print(clf.score(X_test, y_test))
plt.figure(1)
plt.scatter(frame.population,frame.profit,color='red')
plt.xlabel('population')
plt.ylabel('profit')
plt.show

plt.figure(2)
plt.scatter(y_test,y_pred,color='blue')
plt.xlabel('population')
plt.ylabel('profit')
plt.show