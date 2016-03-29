from sklearn import linear_model
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd
loc="student-mat.csv"
data=pd.read_csv(loc,delimiter=';')
matdata=pd.DataFrame(data)

del matdata['famsize']
del matdata['Mjob']
del matdata['Fjob']
del matdata['reason']
matdata.loc[matdata['school']=='GP','school']=1
matdata.loc[matdata['school']=='MS','school']=0

matdata.loc[matdata['sex']=='M','sex']=1
matdata.loc[matdata['sex']=='F','sex']=0

matdata.loc[matdata['address']=='U','address']=1
matdata.loc[matdata['address']=='R','address']=0

matdata.loc[matdata['Pstatus']=='T','Pstatus']=-1
matdata.loc[matdata['Pstatus']=='A','Pstatus']=1

matdata.loc[matdata['romantic']=='yes','romantic']=1
matdata.loc[matdata['romantic']=='no','romantic']=0

matdata.loc[matdata['famsup']=='yes','famsup']=0
matdata.loc[matdata['famsup']=='no','famsup']=1

matdata.loc[matdata['internet']=='yes','internet']=1
matdata.loc[matdata['internet']=='no','internet']=0

matdata.loc[matdata['higher']=='yes','higher']=-1
matdata.loc[matdata['higher']=='no','higher']=1

matdata.loc[matdata['schoolsup']=='yes','schoolsup']=0
matdata.loc[matdata['schoolsup']=='no','schoolsup']=1

matdata.loc[matdata['famsup']=='yes','schoolsup']=0
matdata.loc[matdata['famsup']=='no','schoolsup']=1

matdata.loc[matdata['activities']=='yes','activities']=0
matdata.loc[matdata['activities']=='no','activities']=1

#matdata.loc[matdata['paidclass']=='yes','paidclass']=1
#matdata.loc[matdata['paidclass']=='no','paidclass']=0

matdata.loc[matdata['nursery']=='yes','nursery']=0
matdata.loc[matdata['nursery']=='no','nursery']=1

matdata.loc[matdata['guardian']=='mother','guardian']=-1
matdata.loc[matdata['guardian']=='father','guardian']=0
matdata.loc[matdata['guardian']=='other','guardian']=1


matdata.grade=(matdata.G1+matdata.G2+matdata.G3)/3
grade=pd.DataFrame(matdata.grade,columns=['grade'])
del matdata['G1']
del matdata['G2']
del matdata['G3']
matdata=pd.concat([matdata,grade],axis=1)


matdata.alc=(matdata.Dalc*5+matdata.Walc*2)/7
alc=pd.DataFrame(matdata.alc, columns=['alc'])

del matdata['Dalc']
del matdata['Walc']
matdata=pd.concat([matdata,alc],axis=1)
matdata.loc[matdata['alc']<3,'alc']=0
matdata.loc[matdata['alc']>=3,'alc']=1 

for i in range(len(matdata)):
    if matdata.studytime[i]<2:
        matdata.loc[i,'studytime']=1
    elif matdata.studytime[i]>=2 & matdata.studytime[i]<5:
        matdata.loc[i,'studytime']=2
    elif matdata.studytime[i]>=5 & matdata.studytime[i]<10:
        matdata.loc[i,'studytime']=3
    else:
        matdata.loc[i,'studytime']=4

matdata.tail()
y = matdata.alc
x=matdata.drop(['alc'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

X_train.to_csv("X_train.csv",header=)
X_test.to_csv("X_test.csv")
y_train.to_csv("y_train.csv")
y_test.to_csv("y_test.csv")

from sklearn.ensemble import RandomForestClassifier
feat_labels = X_train.columns
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
importances
