import pandas as pd
from sklearn import preprocessing
import numpy as np
from xgboost import XGBClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn import linear_model

def outlier_detection(x):
    q75, q25 = np.percentile(x, [75 ,25])
    iqr = q75 - q25
    l=q25-iqr
    u=q75+iqr
    return l,u


def XGB_model(train,y):
	model=XGBClassifier(n_estimators=150, learning_rate=0.01)
	from sklearn import cross_validation
	cv = cross_validation.KFold(len(train), n_folds=5,random_state=7)
	for traincv,testcv in cv:
	    model.fit(train.iloc[traincv],y.iloc[traincv])
	y_XGB=model.predict(test)
	return y_XGB

def ADB_model(train,y):
	model2 = AdaBoostClassifier(learning_rate=0.01, n_estimators=300)
	cv = cross_validation.KFold(len(train), n_folds=5,random_state=7)
	for traincv,testcv in cv:
    	model2.fit(train.iloc[traincv],y.iloc[traincv])
    y_ADB=model2.predict(test)
    return y_ADB



def Ensemble(df_XGB,df_ADB):
	frame=pd.DataFrame(data=df_XGB.y,columns=['y'])
	for i in range(0,len(df_ADB)):
	    if (df_ADB.y[i]==1) & (df_XGB.y[i]==1):
	        frame.y[i]=1
	    elif (df_ADB.y[i]==-1) & (df_XGB.y[i]==-1):
	        frame.y[i]=-1
	    elif (df_ADB.y[i]==-1) & (df_XGB.y[i]==1):
	        frame.y[i]=1
	    elif (df_ADB.y[i]==1) & (df_XGB.y[i]==-1):
	        frame.y[i]=1
	return frame


def Save_output(df_ADB,frame2):
	file = open("pred1.txt", "a")
	file.write("Time,Y\n")
	for i in range(0,len(df_ADB)):
	    file.write(str(3000+i))
	    file.write(",")
	    file.write(str(frame2.y[i]))
	    file.write("\n")
    file.close()

if __name__ == '__main__':
	train=pd.read_csv("train.csv")
	test=pd.read_csv("test.csv")
	y=train.Y
	train=train.drop('Y',1)
	train=train.drop('Time',1)
	test=test.drop('Time',1)



	for i in range(0,len(train.iloc[:,0])):
	    l,u=outlier_detection(train.iloc[i,:])
	    for j in range(0,100):
	        if (train.iloc[i,j:j+1]).values[0]>u:
	            train.iloc[i,j:j+1].values[0]=u
	        elif train.iloc[i,j:j+1].values[0]<l:
	            train.iloc[i,j:j+1].values[0]=l
	            
	for i in range(0,len(test.iloc[:,0])):
	    l,u=outlier_detection(test.iloc[i,:])
	    for j in range(0,100):
	        if (test.iloc[i,j:j+1]).values[0]>u:
	            test.iloc[i,j:j+1].values[0]=u
	        elif test.iloc[i,j:j+1].values[0]<l:
	            test.iloc[i,j:j+1].values[0]=l

	for i in range(0,len(train.iloc[:,0])):
	    train.iloc[i,:]=(preprocessing.normalize(train.iloc[i,:], norm='l2'))
	    
	for i in range(0,len(test.iloc[:,0])):
	    test.iloc[i,:]=(preprocessing.normalize(test.iloc[i,:], norm='l2'))

	y_XGB=XGB_model(train,y)
	y_ADB=ADB_model(train,y)
	frame=Ensemble(df_XGB,df_ADB)
	Save_output(df_ADB,frame)