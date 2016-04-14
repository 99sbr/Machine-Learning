import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import svm
df=pd.read_csv('train.csv',delimiter=',')
del df['Loan_ID']
del df['Married']
df['Gender'].fillna("Male",inplace=True)
df['Dependents'].fillna(-1,inplace=True)
df['Self_Employed'].fillna(-1,inplace=True)
df['Credit_History'].fillna(-1,inplace=True)
df.loc[df['Dependents']=='3+','Dependents']=3
df.loc[df['Education']=='Graduate','Education']=1
df.loc[df['Education']=='Not Graduate','Education']=-1
#df.loc[df['Married']=='Yes','Married']=1
#df.loc[df['Married']=='No','Married']=0
df.loc[df['Gender']=='Male','Gender']=1
df.loc[df['Gender']=='Female','Gender']=-1
df.loc[df['Self_Employed']=='Yes','Self_Employed']=1
df.loc[df['Self_Employed']=='No','Self_Employed']=-1
df[df.ApplicantIncome !=0]
df = df[np.isfinite(df['ApplicantIncome'])]
df[df.LoanAmount != 0]
df = df[np.isfinite(df['LoanAmount'])]
df[df.Loan_Amount_Term != 0]
df = df[np.isfinite(df['Loan_Amount_Term'])]
df.loc[df['Loan_Status']=='Y','Loan_Status']=1
df.loc[df['Loan_Status']=='N','Loan_Status']=-1
df.loc[df['Property_Area']=='Urban','Property_Area']=1
df.loc[df['Property_Area']=='Rural','Property_Area']=-1
df.loc[df['Property_Area']=='Semiurban','Property_Area']=0

df.value= (df.ApplicantIncome*df.Loan_Amount_Term)/(df.LoanAmount*1000)
value=pd.DataFrame(df.value,columns=['value_no'])
del df['ApplicantIncome']
del df['CoapplicantIncome']
del df['Loan_Amount_Term']
del df['LoanAmount']
df=pd.concat([df,value],axis=1)
df.to_csv('AV.csv')
av=pd.read_csv('AV.csv',delimiter=',')
y=av.Loan_Status
X=av.drop(['Loan_Status'],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33)
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 15) 
clf = tree.DecisionTreeClassifier()
forest = forest.fit(X_train, y_train)

# Now testing the classifier on test data set
df2=pd.read_csv('test.csv',delimiter=',')
print(df2.shape)
del df2['Married']
del df2['CoapplicantIncome']
df2['Gender'].fillna("Male",inplace=True)
df2['Dependents'].fillna(-1,inplace=True)
df2['Self_Employed'].fillna(-1,inplace=True)
df2['Credit_History'].fillna(-1,inplace=True)
df2.loc[df2['Dependents']=='3+','Dependents']=3
df2.loc[df2['Education']=='Graduate','Education']=1
df2.loc[df2['Education']=='Not Graduate','Education']=-1
#df.loc[df['Married']=='Yes','Married']=1
#df.loc[df['Married']=='No','Married']=0
df2.loc[df2['Gender']=='Male','Gender']=1
df2.loc[df2['Gender']=='Female','Gender']=-1
df2.loc[df2['Self_Employed']=='Yes','Self_Employed']=1
df2.loc[df2['Self_Employed']=='No','Self_Employed']=-1
df2[df2.ApplicantIncome !=0]
df2['ApplicantIncome'].fillna(-1,inplace=True)
#df2 = df2[np.isfinite(df2['ApplicantIncome'])]
df2['LoanAmount'].fillna(-1,inplace=True)
#df2 = df2[np.isfinite(df2['LoanAmount'])]
df2['Loan_Amount_Term'].fillna(-1,inplace=True)
#df2 = df2[np.isfinite(df2['Loan_Amount_Term'])]
df2.loc[df2['Property_Area']=='Urban','Property_Area']=1
df2.loc[df2['Property_Area']=='Rural','Property_Area']=-1
df2.loc[df2['Property_Area']=='Semiurban','Property_Area']=0

df2.value= (df2.ApplicantIncome*df2.Loan_Amount_Term)/(df2.LoanAmount*1000)
value2=pd.DataFrame(df2.value,columns=['value_no'])
del df2['ApplicantIncome']

del df2['Loan_Amount_Term']
del df2['LoanAmount']
df2=pd.concat([df2,value2],axis=1)


print(df2.shape)
df2.to_csv('Sol.csv')
av2=pd.read_csv('Sol.csv',delimiter=',',header=0,quoting=3)
av2=av2.drop(['Loan_ID'],axis=1)

y_pred=forest.predict(av2)
output = pd.DataFrame( data={"Loan_ID": df2["Loan_ID"], "Loan_Status":y_pred} )
output.loc[output['Loan_Status']==-1,'Loan_Status']=0
# Use pandas to write the comma-separated output file
output.to_csv( "Submission.csv", index=False, quoting=3 )