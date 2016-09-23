
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


# In[5]:

import matplotlib.pyplot as plt

import numpy as np
N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radiuses

plt.scatter(frame.age,frame.wage_per_hour, s=frame.income_level, alpha=0.5)
plt.show()


# In[6]:

import seaborn
seaborn.distplot(frame.age, bins=100)
plt.show()


# In[11]:

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(frame['age'],frame['wage_per_hour']) #You can also add more variables here to represent color and size.
plt.show()


# In[18]:

import seaborn as sns
sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")

#df=pd.DataFrame(frame,columns=['age','wage_per_hour','income_level'])
g = sns.lmplot(x="age", y="wage_per_hour", hue="income_level", data=frame,markers=["o", "x"])
plt.title('Histogram')
plt.xlabel('age')
plt.ylabel('wage per hour')
sns.set_style("dark")
plt.show()


# In[14]:

sns.boxplot(x="income_level", y="age", data=frame)
plt.show()


# In[30]:

sns.countplot(x="class_of_worker", hue="income_level",data=frame, palette="Greens_d")
plt.xticks(rotation=90)
plt.show()


# In[29]:

sns.countplot(x="education", hue="income_level",data=frame, palette="Blues_d")
plt.xticks(rotation=90)
plt.show()


# In[ ]:

pd.crosstab(df.A, df.B).apply(lambda r: r/r.sum(), axis=1)

