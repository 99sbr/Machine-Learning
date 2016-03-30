from sklearn import linear_model
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import svm
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

matdata.loc[matdata['paid']=='yes','paid']=1
matdata.loc[matdata['paid']=='no','paid']=0

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


y = matdata.alc
x=matdata.drop(['alc'],axis=1)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.01, random_state=0)
clf = svm.SVC(kernel='linear', C=.001).fit(X_train, y_train)
y_pred=clf.predict(X_test)

print('Coefficients: \n', clf.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((clf.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % clf.score(X_test, y_test))

