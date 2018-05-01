from sklearn import linear_model
from sklearn import cross_validation
from sklearn import svm
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd
loc="student-mat.csv"
loc2="student-por.csv"
data=pd.read_csv(loc,delimiter=';')
data2=pd.read_csv(loc2,delimiter=';')
matdata=pd.DataFrame(data)
pordata=pd.DataFrame(data2)
frames = [matdata,pordata]
result = pd.concat(frames)


del result['famsize']
del result['Mjob']
del result['Fjob']
del result['reason']

result.loc[result['school']=='GP','school']=1
result.loc[result['school']=='MS','school']=0


result.loc[result['sex']=='M','sex']=1
result.loc[result['sex']=='F','sex']=0


result.loc[result['address']=='U','address']=1
result.loc[result['address']=='R','address']=0


result.loc[result['Pstatus']=='T','Pstatus']=-1
result.loc[result['Pstatus']=='A','Pstatus']=1


result.loc[result['romantic']=='yes','romantic']=1
result.loc[result['romantic']=='no','romantic']=0


result.loc[result['famsup']=='yes','famsup']=0
result.loc[result['famsup']=='no','famsup']=1


result.loc[result['internet']=='yes','internet']=1
result.loc[result['internet']=='no','internet']=0


result.loc[result['higher']=='yes','higher']=-1
result.loc[result['higher']=='no','higher']=1


result.loc[result['schoolsup']=='yes','schoolsup']=0
result.loc[result['schoolsup']=='no','schoolsup']=1


result.loc[result['famsup']=='yes','schoolsup']=0
result.loc[result['famsup']=='no','schoolsup']=1


result.loc[result['activities']=='yes','activities']=0
result.loc[result['activities']=='no','activities']=1


result.loc[result['paid']=='yes','paid']=1
result.loc[result['paid']=='no','paid']=0


result.loc[result['nursery']=='yes','nursery']=0
result.loc[result['nursery']=='no','nursery']=1



result.loc[result['guardian']=='mother','guardian']=-1
result.loc[result['guardian']=='father','guardian']=0
result.loc[result['guardian']=='other','guardian']=1



result.grade=(result.G1+result.G2+result.G3)/3
grade=pd.DataFrame(result.grade,columns=['grade'])
del result['G1']
del result['G2']
del result['G3']
result=pd.concat([result,grade],axis=1)



result.alc=(result.Dalc*5+result.Walc*2)/7
alc=pd.DataFrame(result.alc, columns=['alc'])



del result['Dalc']
del result['Walc']
result=pd.concat([result,alc],axis=1)
result.loc[result['alc']<3,'alc']=0
result.loc[result['alc']>=3,'alc']=1 

y = result.alc
x=result.drop(['alc'],axis=1)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=1, random_state=80)
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

print("Residual sum of squares: %.2f"
      % np.mean((clf.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % clf.score(X_test, y_test))