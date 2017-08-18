'''
let us import all necessary library in python

'''
import gc
gc.collect()
import pandas as pd
import numpy as  np
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn import cross_validation
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split




def model_1(X,y,test):
	'''
	This is a catBoost model where we need not to encode categorical variables.
	It automatically takes care of them.
	'''
	categorical_features_indices = np.where(X.dtypes != np.float)[0]
	X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.7, random_state=1234)
	#importing library and building model
	cboost=CatBoostClassifier(iterations=500,learning=0.01,depth=6,loss_function='MultiClass',eval_metric='Accuracy')
	cboost.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_validation, y_validation),plot=True)
	#calculating the class wise prediction probability of cboost model
	pred_prob=cboost.predict_proba(test)
	return pred_prob

def model_2(X,y,test):
	'''
	This is LightGBM.
	'''
	dic={1:0,2:1,3:2}
	y=y.apply(lambda x:dic[x])
	categorical_features_indices = np.where(X.dtypes != np.float)[0]
	categorical=list(train.columns[categorical_features_indices])
	for var in categorical:
		lb = LabelEncoder()
		full_var_data =X.append(test)[var].astype('str')
		temp = lb.fit_transform(np.array(full_var_data))
		X[var] = lb.transform(np.array(X[var]).astype('str'))
		test[var] = lb.transform(np.array(test[var]).astype('str'))

	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1234)
	gbm = lgb.LGBMClassifier(objective='Multiclass',
                        num_leaves=100,
                        learning_rate=0.1,
                        n_estimators=200,num_class=3,max_depth=7,max_bin=200)
	gbm.fit(X,y)
	gbm_prob=gbm.predict_proba(test)
	return gbm_prob

    

def Ensemble(pred1,pred2):
	comb=pd.DataFrame()
	comb['1']=pred1[1]*0.6+pred2['0']*0.4
	comb['2']=pred1[2]*0.6+pred2['1']*0.4
	comb['3']=pred1[3]*0.6+pred2['2']*0.4
	sub=pd.DataFrame()
	sub['Loan_ID']=loan_id
	l=[]
	for i in range(len(comb)):
	    l.append(np.argmax(comb.loc[i]))
	sub['Interest_Rate']=l
	sub.to_csv('comb.csv',index=False)

if __name__ == '__main__':
	print('Loadind datset:\n')
	train=pd.read_csv('train.csv')
	test=pd.read_csv('test.csv')
	print('Train data size %d \n Test data size%d\n'%train.shape,test.shape)
	train=train.drop('Loan_ID',1)
	loan_id=test.Loan_ID
	test=test.drop('Loan_ID',1)
	#Imputing missing values for both train and test
	train.fillna(-999, inplace=True)
	test.fillna(-999,inplace=True)
	#Creating a training set for modeling and validation set to check model performance
	y = train.Interest_Rate
	X = train.drop(['Interest_Rate'], axis=1)

	print('CAT BOOST MODEL RUNNING:\n')
	cat_pred=model_1(X,y,test)
	print('Light GBM MODEL RUNNING:\n')
	gbm_pred=model_2(X,y,test)
	print('Ensembling models')
	Ensemble(cat_pred,gbm_pred)

	
