
# coding: utf-8

# In[474]:

get_ipython().magic('matplotlib inline')
import pandas as pd
data=pd.read_csv('ign.csv')
data=pd.DataFrame(data)


# In[475]:

data.head()


# In[476]:

ECA=data.groupby('platform')
ECA2=data.groupby('platform').release_year


# In[477]:

import numpy as np


# In[478]:

max_dict={}


# In[479]:

for name , group in ECA:
    max_dict[name] = np.array(list(group["editors_choice"].value_counts()))[1:]


# In[480]:

get_ipython().magic('matplotlib inline')
print("list of platforms in decreasing order of editors choice award:")
print(max_dict.keys())


# In[481]:

trace1 = go.Scatter(
    y = list(max_dict.values()),
    x =list(max_dict.keys()),
    mode = 'lines+markers',
    name='Editor Choice Award'
)


gdata = [trace1]

# Plot and embed in ipython notebook!
py.iplot(gdata, filename='Editors_choice',title="Most Editor choice Award")


# In[ ]:




# In[482]:

max_dict2={}
for name,group in ECA:
    max_dict2[name]=max(np.array(list(group["release_year"].value_counts())))


# In[ ]:




# In[483]:

import plotly.plotly as py
import plotly.graph_objs as go

# Create random data with numpy
import numpy as np

# Create a trace
trace1 = go.Scatter(
    y = list(max_dict.values()),
    x =list(max_dict.keys()),
    mode = 'lines+markers'
)
trace2 = go.Scatter(
    y = list(max_dict2.values()),
    x =list(max_dict2.keys()),
    mode = 'lines+markers'
)


gdata = [trace1,trace2]

# Plot and embed in ipython notebook!
py.iplot(gdata, filename='release_editors_reln')


# In[484]:

'''
The graph clearly shows that whenever any platform has showed hike in number of release in a particular
year the editors choice winning probability has increased.
'''


# In[ ]:




# In[485]:

x=data.groupby('platform').release_year.value_counts()


# In[ ]:




# In[486]:

Mac=x['Macintosh']


# In[487]:

Mac


# In[488]:

Mac_average_awd_count=np.mean(Mac)


# In[489]:

Mac_average_awd_count


# In[490]:

max_dict3={}
a=data.groupby('release_month').editors_choice.value_counts()
a


# In[491]:

l=int(len(a)/2)
for i in range(0,l):
    max_dict3[i]=a.iloc[i+1]
max_dict3


# In[ ]:




# In[492]:

import plotly.plotly as py
import plotly.graph_objs as go

# Create random data with numpy
import numpy as np

# Create a trace
trace = go.Scatter(
    y = list(max_dict3.values()),
    x =list(max_dict3.keys()),
    mode = 'lines+markers'
)

gdata = [trace]

# Plot and embed in ipython notebook!
py.iplot(gdata, filename='optimal_month')


# In[493]:

year=data.release_year


# In[494]:

unique, counts = np.unique(np.asarray(year), return_counts=True)


# In[495]:

print(unique,counts)
def growth(array):
    l=len(array)
    per_growth=[]
    per_growth.append(0)
    for  i in range(1,l):
        per_growth.append(((array[i]-array[i-1])/array[i-1])*100)
    return per_growth
per_growth=[]
per_growth=growth(counts)


# In[496]:

import plotly.plotly as py
import plotly.graph_objs as go


trace=go.Pie(labels=unique,values=per_growth[2:])

py.iplot([trace],title='Per growth excluding first 2 years')


# In[497]:

trace = go.Scatter(
    y = unique,
    x =per_growth,
    mode = 'lines+markers'
)

gdata = [trace]

# Plot and embed in ipython notebook!
py.iplot(gdata, filename='growth trend')


# In[498]:

trace = go.Scatter(
    x = unique,
    y =per_growth[2:],
    mode = 'lines+markers'
)

gdata = [trace]

# Plot and embed in ipython notebook!
py.iplot(gdata, filename='growth trend excluding first 2 years')


# In[ ]:




# In[ ]:




# In[ ]:



