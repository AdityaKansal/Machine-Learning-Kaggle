# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 10:15:40 2018

@author: akansal2
"""

#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



#Getting the data set
Dataset = pd.read_csv('C:\\A_stuff\\Learning\\Machine Learning\\Kaggle\\Titanic Problem\\train.csv')
Dataset_test = pd.read_csv('C:\\A_stuff\\Learning\\Machine Learning\\Kaggle\\Titanic Problem\\test.csv')

#look at the data
Dataset.head(2)
Dataset.info()


#Deleting irrevalent columns
Dataset.drop(['PassengerId','Name','Ticket','Fare','Cabin'],axis = 1,inplace = True)
Dataset_test.drop(['PassengerId','Name','Ticket','Fare','Cabin'],axis = 1,inplace = True)

##Adding parch and sibsp
#temp = Dataset['SibSp'] + Dataset['Parch']
#Dataset = pd.concat([Dataset,temp.rename('Dependent')],axis = 1)
#Dataset.drop(['Parch','SibSp'],axis = 1,inplace = True)

#checking missing values
pd.isna(Dataset).sum()
pd.isna(Dataset_test).sum()


#finding out meean and mode for those columns
int(Dataset['Age'].mean())
int(Dataset['Age'].mode()[0])
str(Dataset['Embarked'].mode()[0])

#filling missing values
Dataset['Age'].fillna(int(Dataset['Age'].mode()[0]),inplace = True)
Dataset['Embarked'].fillna(str(Dataset['Embarked'].mode()[0]),inplace = True)
Dataset_test['Age'].fillna(int(Dataset_test['Age'].mode()[0]),inplace = True)

#checking again missing values
pd.isna(Dataset).sum()
Dataset.head(2)


#decribing data
Dataset.describe()



#identifying categorical variables for label encoder
CV_labelEncoder = ['Sex','Embarked']


from sklearn.preprocessing import LabelEncoder
for i in CV_labelEncoder:
    le = LabelEncoder()
    temp = le.fit_transform(Dataset[i].values)
    Dataset[i] = pd.Series(temp)
    Dataset_test[i] = le.transform(Dataset_test[i].values)





#identifying categorical variables for one hot encoding
CV_Dummies = ['Pclass','Sex','Embarked']

for i in CV_Dummies:  
      temp = pd.get_dummies(Dataset[i],drop_first = True,prefix = i + '_dummies')
      Dataset = pd.concat([Dataset,temp],axis= 1)
      Dataset.drop([i],axis =1,inplace = True)
      temp = pd.get_dummies(Dataset_test[i],drop_first = True,prefix = i + '_dummies')
      Dataset_test = pd.concat([Dataset_test,temp],axis= 1)
      Dataset_test.drop([i],axis =1,inplace = True)





#dividing data into matrices
X = np.array(Dataset.drop(['Survived'],axis =1))
y =np.array(Dataset['Survived'])




#splitting data into train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)



##featue scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)


#
#
##fitting classifier
#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(n_estimators = 10,criterion = 'entropy',random_state = 0)
#classifier.fit(X_train,y_train)

#
##fitting SVM rbf kernel classifier
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf' , random_state= 0)
classifier.fit(X_train,y_train)


##fitting  SVM poly classifier
#from sklearn.svm import SVC
#classifier = SVC(kernel = 'poly' ,degree= 3, random_state= 0)
#classifier.fit(X_train,y_train)


##fitting KNN
#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors= 5,p=2,metric = 'minkowski')
#classifier.fit(X_train,y_train)

#predicting y_pred
y_pred = classifier.predict(X_test)


#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#K-FOLD
from sklearn.model_selection import cross_val_score
cvs = cross_val_score(estimator = classifier, X = X_train,y = y_train,cv = 10)
cvs.mean()




#predicting actual test
X_actutaltest = np.array(Dataset_test)
X_actutaltest = sc.transform(X_actutaltest)
y_pred2 = classifier.predict(X_actutaltest))


A = pd.DataFrame(y_pred2)

from openpyxl import load_workbook
writer = pd.ExcelWriter('C:\\A_stuff\\Learning\\Machine Learning\\Kaggle\\Titanic Problem\\Output.xls')
A.to_excel(writer,'Sheet2')
writer.save()




































































