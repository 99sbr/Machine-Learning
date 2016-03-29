from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
loc="student-mat.csv"
data=pd.read_csv(loc,delimiter=';')
matdata=pd.DataFrame(data)
matdata.loc[matdata['school']=='GP','school']=1
matdata.loc[matdata['school']=='MS','school']=0
matdata.loc[matdata['sex']=='M','sex']=1
matdata.loc[matdata['sex']=='F','sex']=0
matdata.loc[matdata['address']=='U','address']=1
matdata.loc[matdata['address']=='R','address']=0
matdata.loc[matdata['Pstatus']=='T','Pstatus']=1
matdata.loc[matdata['Pstatus']=='A','Pstatus']=0
matdata.loc[matdata['romantic']=='yes','romantic']=1
matdata.loc[matdata['romantic']=='no','romantic']=0
matdata.alc=(matdata.Dalc*5+matdata.Walc*2)/7
alc=pd.DataFrame(matdata.alc, columns=['alc'])

del matdata['Dalc']
del matdata['Walc']
matdata=pd.concat([matdata,alc],axis=1)
matdata
