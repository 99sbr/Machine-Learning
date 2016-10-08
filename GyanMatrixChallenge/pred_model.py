
# coding: utf-8

# In[152]:

import pandas as pd
data=pd.read_csv('ign.csv')
data=pd.DataFrame(data)
data=data.drop(['Unnamed: 0','title','url'],1)


# In[153]:

data.head()


# In[154]:

from sklearn.preprocessing import LabelEncoder
import numpy as np
categorical=['score_phrase','platform','genre','editors_choice']


# In[155]:

for var in categorical:
    lb = LabelEncoder()
    full_var_data =data[var].astype('str')
    temp = lb.fit_transform(np.array(full_var_data))
    data[var] = lb.transform(np.array(data[var] ).astype('str'))
   


# In[156]:

data.head()


# In[157]:

print(data.editors_choice.value_counts())
print(data.release_month.value_counts())


# In[158]:

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


# In[159]:

print("Getting count features..")
data["platform_Count"] = getCountVar(data, data,"platform")
data["platform_Count"] = getCountVar(data,data, "platform")
print ("Platform", np.unique(data["platform_Count"]))

data["release_month_Count"] = getCountVar(data, data,"release_month")
data["release_month_Count"] = getCountVar(data,data, "release_month")
print ("release_month", np.unique(data["release_month_Count"]))


# In[160]:

data.head()


# In[161]:

from sklearn.cross_validation import train_test_split
from sklearn import cross_validation

train,test=train_test_split(data,train_size=0.8)
y_train=train.editors_choice
y_test=test.editors_choice

X_train=train.drop('editors_choice',1)
X_test=test.drop('editors_choice',1)

from sklearn.linear_model import LogisticRegression
cv = cross_validation.KFold(len(X_train), n_folds=5)


# In[ ]:




# In[ ]:




# In[162]:

from sklearn.metrics import accuracy_score


# In[ ]:




# In[163]:

from sklearn.metrics import confusion_matrix


# In[ ]:




# In[ ]:


            


# In[ ]:




# In[164]:

from xgboost import XGBClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
y=data.editors_choice
X=data.drop('editors_choice',1)


# In[165]:


model = XGBClassifier()
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
param_grid = dict(learning_rate=learning_rate)
kfold = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="f1", n_jobs=-1, cv=kfold)
result = grid_search.fit(X,y)
# summarize results
print("Best: %f using %s" % (result.best_score_, result.best_params_))
means, stdevs = [], []
for params, mean_score, scores in result.grid_scores_:
    stdev = scores.std()
    means.append(mean_score)
    stdevs.append(stdev)
    print("%f (%f) with: %r" % (mean_score, stdev, params))


# In[166]:

model=XGBClassifier(learning_rate=0.2)


# In[167]:

for traincv, testcv in cv:
    model.fit(X_train.iloc[traincv],y_train.iloc[traincv])


# In[168]:

y_XGB=model.predict(X_test)


# In[169]:

accuracy_score(y_test, y_XGB)


# In[170]:

from sklearn.metrics import confusion_matrix
tn, fp, fn, tp=confusion_matrix(y_test, y_XGB).ravel()


from sklearn.metrics import classification_report

target_names = ['N', 'Y']
print(classification_report(y_test, y_XGB, target_names=target_names))
            


# In[171]:

from xgboost import plot_importance
from matplotlib import pyplot
print(model.feature_importances_)


# In[172]:

get_ipython().magic('matplotlib inline')

import matplotlib
plot_importance(model)
pyplot.show()


# In[ ]:




# In[ ]:



