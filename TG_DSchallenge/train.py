
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.preprocessing import LabelEncoder
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')


# In[191]:

cat_columns=['city', 'state', 'store_location', 'time_zone',
       'location_employee_code', 'credit_score']


# In[192]:

train['credit_score_range_max']=0
train['credit_score_range_min']=0


test['credit_score_range_max']=0
test['credit_score_range_min']=0


for i in range(len(train)):
    try:
        train['credit_score_range_max'][i]=int(train.credit_score_range[i].split('TO')[1])
        train['credit_score_range_min'][i]=int(train.credit_score_range[i].split('TO')[0])
        train['credit_score_range'][i]=int(int(train.credit_score_range[i].split('TO')[1])-int(train.credit_score_range[i].split('TO')[0]))
    except:
        train['credit_score_range'][i]=0
        train['credit_score_range_max'][i]=0
        train['credit_score_range_min'][i]=0
    
    
for i in range(len(test)):
    try:
        test['credit_score_range_max'][i]=int(test.credit_score_range[i].split('TO')[1])
        test['credit_score_range_min'][i]=int(test.credit_score_range[i].split('TO')[0])
        test['credit_score_range'][i]=int(int(test.credit_score_range[i].split('TO')[1])-int(test.credit_score_range[i].split('TO')[0]))
    except:
        test['credit_score_range'][i]=0
        test['credit_score_range_max'][i]=0
        test['credit_score_range_min'][i]=0
    


# In[193]:

for var in cat_columns:
                lb = LabelEncoder()
                full_var_data = pd.concat((train[var],test[var]),axis=0).astype('str')
                temp = lb.fit_transform(np.array(full_var_data))
                train[var] = lb.transform(np.array( train[var] ).astype('str'))
                test[var] = lb.transform(np.array( test[var] ).astype('str'))


# In[194]:

train['credit_score_range'] = train['credit_score_range'].apply(pd.to_numeric)
test['credit_score_range'] = test['credit_score_range'].apply(pd.to_numeric)


# In[195]:

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


# In[196]:

#store_location
train['store_location_Count']=getCountVar(train,train,'store_location')
test['store_location_Count']=getCountVar(test,train,'store_location')
#time_zone
train['time_zone_Count']=getCountVar(train,train,'time_zone')
test['time_zone_Count']=getCountVar(test,train,'time_zone')
#location_employee_code
train['location_employee_code_Count']=getCountVar(train,train,'location_employee_code')
test['location_employee_code_Count']=getCountVar(test,train,'location_employee_code')


# In[197]:

train['normalized_household_income']=(train['total_household_income']/train['employee_size'])
test['normalized_household_income']=(test['total_household_income']/test['employee_size'])
train=train.drop('total_household_income',1)
test=test.drop('total_household_income',1)


# In[198]:

y=train.total_sales
train=train.drop(['total_sales','outlet_no'],1)
outlet=test.outlet_no
test=test.drop('outlet_no',1)


# In[199]:

from xgboost import XGBRegressor
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV


# In[209]:

model = XGBRegressor()
learning_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
n_estimators=[100,200,300,400,500]
param_grid = dict(learning_rate=learning_rate,n_estimators=n_estimators)
kfold = StratifiedKFold(y, n_folds=3, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="mean_absolute_error", n_jobs=-1, cv=kfold)


# In[210]:

result = grid_search.fit(train,y)
# summarize results
print("Best: %f using %s" % (result.best_score_, result.best_params_))


# In[211]:

model=XGBRegressor(learning_rate=0.3,n_estimators=100)
for traincv,testcv in kfold:
    model.fit(train.iloc[traincv],y.iloc[testcv])



# In[212]:

y_pred=model.predict(test)


# In[213]:

output2 = pd.DataFrame( data={"outlet_no":outlet,"total_sales_Actual": y_pred} )
output2.to_csv("model.csv", index=False,quoting=3)


# In[ ]:



