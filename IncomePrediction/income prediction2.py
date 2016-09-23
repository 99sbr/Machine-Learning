
# coding: utf-8

# In[1]:

from azureml import Workspace

ws = Workspace()
ds = ws.datasets['train.csv']
frame = ds.to_dataframe()


# In[2]:

frame.loc[frame['income_level']==-50000,'income_level']=0
frame.loc[frame['income_level']==50000,'income_level']=1
frame.income_level.head()


# In[3]:

cat_frame=frame.select_dtypes(include=['object'])
num_frame=frame.select_dtypes(exclude=['object'])


# In[5]:

import seaborn
import matplotlib.pyplot as plt
seaborn.distplot(num_frame.age, bins=100)
plt.show()


# In[6]:

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(num_frame['age'],num_frame['wage_per_hour']) #You can also add more variables here to represent color and size.
plt.show()


# In[7]:

import seaborn as sns
sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")

#df=pd.DataFrame(frame,columns=['age','wage_per_hour','income_level'])
g = sns.lmplot(x="age", y="wage_per_hour", hue="income_level", data=num_frame,markers=["o", "x"])
plt.title('Histogram')
plt.xlabel('age')
plt.ylabel('wage per hour')
plt.show()


# In[8]:

sns.boxplot(x="income_level", y="age", data=num_frame)
plt.show()


# In[11]:

sns.countplot(x="class_of_worker", hue="income_level",data=frame, palette="Greens_d")
plt.xticks(rotation=90)
plt.show()


# In[12]:

sns.countplot(x="education", hue="income_level",data=frame, palette="Blues_d")
plt.xticks(rotation=90)
plt.show()


# In[13]:

corrMatrix=num_frame.corr()
corrMatrix


# In[16]:

import numpy as np
corrMatrix.loc[:,:] =  np.tril(corrMatrix, k=-1) # borrowed from Karl D's answer
already_in = set()
result = []
for col in corrMatrix:
    perfect_corr = corrMatrix[col][corrMatrix[col] > 0.5].index.tolist()
    if perfect_corr and col not in already_in:
        already_in.update(set(perfect_corr))
        perfect_corr.append(col)
        result.append(perfect_corr)
result


# In[18]:

del num_frame['veterans_benefits']
del num_frame['weeks_worked_in_year']


# In[19]:

def num_missing(x):
    return (sum(x.isnull())/len(x))*100

#Applying per column:
print("Missing values per column:")
print(cat_frame.apply(num_missing, axis=0)) #axis=0 defines that function is to be applied on each column


# In[21]:

# deleting columns whose missing values ~50%
del cat_frame['migration_msa'] 
del cat_frame['migration_reg']
del cat_frame['migration_sunbelt']
del cat_frame['migration_within_reg']


# In[22]:

cat_frame['hispanic_origin'].fillna('Unavailable',inplace=True)
cat_frame['state_of_previous_residence'].fillna('Unavailable',inplace=True)
cat_frame['country_father'].fillna('Unavailable',inplace=True)
cat_frame['country_mother'].fillna('Unavailable',inplace=True)
cat_frame['country_self'].fillna('Unavailable',inplace=True)


# In[23]:

print(cat_frame.apply(num_missing, axis=0))
# No null values in  categorical variables now


# In[34]:

import pandas as pd
print(pd.value_counts(num_frame['age'].values, sort=False))


# In[42]:

dummy1=pd.concat([num_frame['age'],num_frame['income_level']],axis=1)
print(dummy1.groupby(["income_level"], as_index=True).count())
dummy2=pd.concat([num_frame['wage_per_hour'],num_frame['income_level']],axis=1)
print(dummy2.groupby(["income_level"], as_index=True).count())


# In[ ]:



