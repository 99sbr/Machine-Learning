from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
import  numpy as np
import pandas as pd
from datetime import datetime
import datetime as dt
import xgboost as xgb
from sklearn import cross_validation

def getVar(compute_df, purchase_df, var_name):
        grouped_df = purchase_df.groupby(var_name)
        min_dict = {}
        max_dict = {}
        mean_dict = {}
        twentyfive_dict = {}
        seventyfive_dict = {}
        for name, group in grouped_df:
                min_dict[name] = min(np.array(group["Footfall"]))
                max_dict[name] = max(np.array(group["Footfall"]))
                mean_dict[name] = np.mean(np.array(group["Footfall"]))
                twentyfive_dict[name] = np.percentile(np.array(group["Footfall"]),25)
                seventyfive_dict[name] = np.percentile(np.array(group["Footfall"]),75)

        min_list = []
        max_list = []
        mean_list = []
        twentyfive_list = []
        seventyfive_list = []
        for index, row in compute_df.iterrows():
                name = row[var_name]
                min_list.append(min_dict.get(name,0))
                max_list.append(max_dict.get(name,0))
                mean_list.append(mean_dict.get(name,0))
                twentyfive_list.append( twentyfive_dict.get(name,0))
                seventyfive_list.append( seventyfive_dict.get(name,0))

        return min_list, max_list, mean_list, twentyfive_list, seventyfive_list
def getCountVar(compute_df, count_df, var_name):
        grouped_df = count_df.groupby(var_name)
        count_dict = {}
        for name, group in grouped_df:
                count_dict[name] = group.shape[0]

        count_list = []
        for index, row in compute_df.iterrows():
                name = row[var_name]
                count_list.append(count_dict.get(name, 0))
        return count_list
import numpy as np
def outlier_detection(x):
    q75, q25 = np.percentile(x, [75 ,25])
    iqr = q75 - q25
    l=q25
    u=q75
    return l,u
def model_1():
	print("Computing Model 1")
	train_data = pd.read_csv('Train_xyqdbho.csv')
	test_data = pd.read_csv('Test_pyI9Owa.csv')
	train_data['Year'] = pd.DatetimeIndex(train_data['Date']).year
	train_data['Month'] = pd.DatetimeIndex(train_data['Date']).month
	train_data['Day'] = pd.DatetimeIndex(train_data['Date']).day
	del train_data['Date']

	train_data.fillna(0,inplace=True)
	#####################################
	test_data['Year'] = pd.DatetimeIndex(test_data['Date']).year
	test_data['Month'] = pd.DatetimeIndex(test_data['Date']).month
	test_data['Day'] = pd.DatetimeIndex(test_data['Date']).day
	del test_data['Date']
	test_data.fillna(0,inplace=True)
	X_test=test_data
	y_train=train_data.Footfall
	X_train=train_data
	yID=pd.DataFrame(X_test.ID)
	yID.head()
	X_train.head()
	#X_train=X_train.drop('Park_ID',1)
	X_train= X_train.drop('ID',1)
	#X_test=X_test.drop('Park_ID',1)
	X_test= X_test.drop('ID',1)
	print("Getting count features..")
	X_train["Month_Count"] = getCountVar(X_train, X_train,"Month")
	X_test["Month_Count"] = getCountVar(X_test, X_train, "Month")
	print ("Months", np.unique(X_test["Month_Count"]))

	X_train.head()
	print("Getting count features..")
	X_train["Day_Count"] = getCountVar(X_train, X_train,"Day")
	X_test["Day_Count"] = getCountVar(X_test, X_train, "Day")
	print ("Dyss", np.unique(X_test["Day_Count"]))

	X_train.head()
	
	X_train["Loc_Count"] = getCountVar(X_train, X_train,"Location_Type")
	X_test["Loc_Count"] = getCountVar(X_test, X_train, "Location_Type")

	Var1_min_list, Var1_max_list, Var1_mean_list, Var1_25_list, Var1_75_list=getVar(X_train,X_train,"Var1")
	X_train['Var1_min_list']=(Var1_min_list)
	X_train['Var1_max_list']=Var1_max_list
	X_train['Var1_mean_list']=(Var1_mean_list)
	X_train['Var1_25_list']=(Var1_25_list)
	X_train['Var1_75_list']=(Var1_75_list)
	Var1_min_list, Var1_max_list, Var1_mean_list, Var1_25_list, Var1_75_list=getVar(X_test,X_train,"Var1")
	X_test['Var1_min_list']=(Var1_min_list)
	X_test['Var1_max_list']=(Var1_max_list)
	X_test['Var1_mean_list']=(Var1_mean_list)
	X_test['Var1_25_list']=(Var1_25_list)
	X_test['Var1_75_list']=(Var1_75_list)
	DOW_min_list,DOW_max_list, DOW_mean_list, DOW_25_list, DOW_75_list=getVar(X_train,X_train,"Direction_Of_Wind")
	X_train['Direction_Of_Wind_min_list']=(DOW_min_list)
	X_train['Direction_Of_Wind_max_list']=(DOW_max_list)
	X_train['Direction_Of_Wind_mean_list']=(DOW_mean_list)
	X_train['Direction_Of_Wind_25_list']=(DOW_25_list)
	X_train['Direction_Of_Wind_75_list']=(DOW_75_list)

	DOW_min_list, DOW_max_list, DOW_mean_list, DOW_25_list, DOW_75_list=getVar(X_test,X_train,"Direction_Of_Wind")
	X_test['Direction_Of_Wind_min_list']=(DOW_min_list)
	X_test['Direction_Of_Wind_max_list']=(DOW_max_list)
	X_test['Direction_Of_Wind_mean_list']=(DOW_mean_list)
	X_test['Direction_Of_Wind_25_list']=(DOW_25_list)
	X_test['Direction_Of_Wind_75_list']=(DOW_75_list)

	X_train=X_train.drop('Footfall',1)
	X_train['MD']=X_train['Month']*X_train['Day']*X_train['Location_Type']
	X_test['MD']=X_test['Month']*X_test['Day']*X_test['Location_Type']
	del X_train['Year']
	del X_test['Year']
	xgb1 = xgb.XGBRegressor(
		 learning_rate =0.05,
		 n_estimators=300,
		 max_depth=10,
		 min_child_weight=1,
		 gamma=0,
		   
		 subsample=0.8,
		 colsample_bytree=0.8,
		 nthread=4,
		 scale_pos_weight=1,
		 seed=27)

	cv = cross_validation.KFold(len(X_train), n_folds=5)
	print("Training Model 1")
	for traincv, testcv in cv:
		xgb1.fit(X_train.iloc[traincv], y_train.iloc[traincv],eval_metric='rmse')
	y=xgb1.predict(X_test)
	output2 = pd.DataFrame( data={"Footfall":y,"ID": yID["ID" ]} )
	output2.to_csv("XGBregressor.csv", index=False,quoting=3)



def model_2():
	print("Computing Model 2")
	train_data = pd.read_csv('Train_xyqdbho.csv')
	test_data = pd.read_csv('Test_pyI9Owa.csv')
	train_data['Year'] = pd.DatetimeIndex(train_data['Date']).year
	train_data['Month'] = pd.DatetimeIndex(train_data['Date']).month
	train_data['Day'] = pd.DatetimeIndex(train_data['Date']).day
	del train_data['Date']

	train_data.fillna(0,inplace=True)
	#####################################
	test_data['Year'] = pd.DatetimeIndex(test_data['Date']).year
	test_data['Month'] = pd.DatetimeIndex(test_data['Date']).month
	test_data['Day'] = pd.DatetimeIndex(test_data['Date']).day
	del test_data['Date']

	test_data.fillna(0,inplace=True)
	X_test=test_data
	y_train=train_data.Footfall
	X_train=train_data
	yID=pd.DataFrame(X_test.ID)
	yID.head()
	
	#X_train=X_train.drop('Park_ID',1)
	X_train= X_train.drop('ID',1)
	#X_test=X_test.drop('Park_ID',1)
	X_test= X_test.drop('ID',1)
	X_train= X_train.drop('Year',1)
	X_test=X_test.drop('Year',1)
	X_train['Breeze_Speed']= (X_train['Max_Breeze_Speed']-X_train['Min_Breeze_Speed'])/X_train['Average_Breeze_Speed']
	del X_train['Max_Breeze_Speed']
	del X_train['Min_Breeze_Speed']
	del X_train['Average_Breeze_Speed']
	X_train['Atmospheric_Pressure']= (X_train['Max_Atmospheric_Pressure']-X_train['Min_Atmospheric_Pressure'])/X_train['Average_Atmospheric_Pressure']
	del X_train['Max_Atmospheric_Pressure']
	del X_train['Min_Atmospheric_Pressure']
	del X_train['Average_Atmospheric_Pressure']
	X_train['Ambient_Pollution']= (X_train['Min_Ambient_Pollution']/X_train['Max_Ambient_Pollution'])
	del X_train['Max_Ambient_Pollution']
	del X_train['Min_Ambient_Pollution']
	X_train['Moisture_In_Park']= (X_train['Max_Moisture_In_Park']-X_train['Min_Moisture_In_Park'])/X_train['Average_Moisture_In_Park']
	del X_train['Max_Moisture_In_Park']
	del X_train['Min_Moisture_In_Park']
	del X_train['Average_Moisture_In_Park']
	#del X_test['Day']
	#del X_test['Year']
	#del X_test['Direction_Of_Wind']
	X_test['Breeze_Speed']= (X_test['Max_Breeze_Speed']-X_test['Min_Breeze_Speed'])/X_test['Average_Breeze_Speed']
	del X_test['Max_Breeze_Speed']
	del X_test['Min_Breeze_Speed']
	del X_test['Average_Breeze_Speed']
	X_test['Atmospheric_Pressure']= (X_test['Max_Atmospheric_Pressure']-X_test['Min_Atmospheric_Pressure'])/X_test['Average_Atmospheric_Pressure']
	del X_test['Max_Atmospheric_Pressure']
	del X_test['Min_Atmospheric_Pressure']
	del X_test['Average_Atmospheric_Pressure']
	X_test['Ambient_Pollution']= (X_test['Min_Ambient_Pollution']/X_test['Max_Ambient_Pollution'])
	del X_test['Max_Ambient_Pollution']
	del X_test['Min_Ambient_Pollution']
	X_test['Moisture_In_Park']= (X_test['Max_Moisture_In_Park']-X_test['Min_Moisture_In_Park'])/X_test['Average_Moisture_In_Park']
	del X_test['Max_Moisture_In_Park']
	del X_test['Min_Moisture_In_Park']
	del X_test['Average_Moisture_In_Park']
	X_train['Direction_Of_Wind']=X_train['Direction_Of_Wind']*(3.14/180)
	X_test['Direction_Of_Wind']=X_test['Direction_Of_Wind']*(3.14/180)
	Var1_min_list, Var1_max_list, Var1_mean_list, Var1_25_list, Var1_75_list=getVar(X_train,X_train,"Var1")
	X_train['Var1_min_list']=np.log(Var1_min_list)
	X_train['Var1_max_list']=np.log(Var1_max_list)
	X_train['Var1_mean_list']=np.log(Var1_mean_list)
	X_train['Var1_25_list']=np.log(Var1_25_list)
	X_train['Var1_75_list']=np.log(Var1_75_list)
	Var1_min_list, Var1_max_list, Var1_mean_list, Var1_25_list, Var1_75_list=getVar(X_test,X_train,"Var1")
	X_test['Var1_min_list']=np.log(Var1_min_list)
	X_test['Var1_max_list']=np.log(Var1_max_list)
	X_test['Var1_mean_list']=np.log(Var1_mean_list)
	X_test['Var1_25_list']=np.log(Var1_25_list)
	X_test['Var1_75_list']=np.log(Var1_75_list)
	DOW_min_list,DOW_max_list, DOW_mean_list, DOW_25_list, DOW_75_list=getVar(X_train,X_train,"Direction_Of_Wind")
	X_train['Direction_Of_Wind_min_list']=np.log(DOW_min_list)
	X_train['Direction_Of_Wind_max_list']=np.log(DOW_max_list)
	X_train['Direction_Of_Wind_mean_list']=np.log(DOW_mean_list)
	X_train['Direction_Of_Wind_25_list']=np.log(DOW_25_list)
	X_train['Direction_Of_Wind_75_list']=np.log(DOW_75_list)

	DOW_min_list, DOW_max_list, DOW_mean_list, DOW_25_list, DOW_75_list=getVar(X_test,X_train,"Direction_Of_Wind")
	X_test['Direction_Of_Wind_min_list']=np.log(DOW_min_list)
	X_test['Direction_Of_Wind_max_list']=np.log(DOW_max_list)
	X_test['Direction_Of_Wind_mean_list']=np.log(DOW_mean_list)
	X_test['Direction_Of_Wind_25_list']=np.log(DOW_25_list)
	X_test['Direction_Of_Wind_75_list']=np.log(DOW_75_list)

	Location_type=pd.DataFrame(X_train.Location_Type)
	month=pd.DataFrame(X_train.Month)
	footfall=pd.DataFrame(X_train.Footfall)
	data=pd.concat([Location_type,month,footfall],1)
	matrix = data.pivot_table(index=['Location_Type'], columns=['Month'], values='Footfall')
	
	matrix.iloc[:,[0]]
	loc1=outlier_detection(matrix[0:][0:1])
	loc2=outlier_detection(matrix[0:][1:2])
	loc3=outlier_detection(matrix[0:][2:3])
	loc4=outlier_detection(matrix[0:][3:4])
	loc=[loc1,loc2,loc3,loc4]
	a,b=matrix.shape
	a,b=matrix.shape
	d={}
	for i in range(0,a):
	    a=[]
	    arr=np.asarray(matrix.iloc[i])
	    for j in  range(0,b):
	        if np.asarray(arr[j]) >=loc[i][1] or arr[j]<=loc[i][0]:
	            a.append(j)
	    d[i]=a


	infected_months=[0,1,5,6,7,11]

	X_train.loc[(X_train['Month']==0) | (X_train['Month']==1) | (X_train['Month']==5) | (X_train['Month']==6 )
	                           |
	            (X_train['Month']==7) | (X_train['Month']==11),'flag']=1



	X_train.flag.fillna(0,inplace=True)
	X_test.loc[(X_test['Month']==0) | (X_test['Month']==1) | (X_test['Month']==5) | (X_test['Month']==6 )
	                           |
	            (X_test['Month']==7) | (X_test['Month']==11),'flag']=1



	X_test.flag.fillna(0,inplace=True)
	X_train=X_train.drop('Footfall',1)
	X_train.Var1=np.log(X_train.Var1)

	X_test.Var1=np.log(X_test.Var1)
	X_train['MD']=(X_train['Month']*X_train['Day'])/X_train['Location_Type']
	X_test['MD']=(X_test['Month']*X_test['Day'])/X_test['Location_Type']

	xgb1 = xgb.XGBRegressor(
	 learning_rate =0.05,
	 n_estimators=300,
	 max_depth=10,
	 min_child_weight=1,
	 gamma=0,
	 subsample=0.8,
	 colsample_bytree=0.8,
	 nthread=4,
	 scale_pos_weight=1,
	 seed=1)
	cv = cross_validation.KFold(len(X_train), n_folds=5)
	print("Training Model 2")
	for traincv, testcv in cv:
	    xgb1.fit(X_train.iloc[traincv], y_train.iloc[traincv],eval_metric='rmse')

	y=xgb1.predict(X_test)
	output= pd.DataFrame( data={"ID": yID["ID"], "Footfall":y} )
	output.to_csv("XGB.csv", index=False,quoting=3)







if __name__ == '__main__':
	model_1()
	model_2()
	output1=pd.read_csv('XGB.csv')
	output2=pd.read_csv('XGBregressor.csv')
	for i in range(0,2):
		output2['Footfall']=0.7*output2['Footfall']+0.3*output1['Footfall']
	output2.to_csv("Submission.csv",index=False,quoting=3)