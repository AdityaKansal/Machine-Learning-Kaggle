# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 17:07:25 2018

@author: akansal2
"""



#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(threshold = 100)     #to increase the view of ndarray

#importing data set
Dataset = pd.read_csv('C:\\A_stuff\\Learning\\Machine Learning\\Kaggle\\Titanic Problem\\train.csv')
#Dataset = pd.read_csv('C:\\A_stuff\\Learning\\Machine Learning\\Kaggle\\Titanic Problem\\test.csv')


#checking head of dataset
print(Dataset.head(2))


#deleting unncessary data
Dataset =Dataset.drop(['PassengerId','Name','Ticket','Cabin'],axis =1)



#checking missing values
print(Dataset.iloc[:,:].isnull().sum())



#filling missing values
Dataset.iloc[:,3] = Dataset.iloc[:,3].fillna(str(Dataset.iloc[:,3].mean()))
Dataset.iloc[:,7] = Dataset.iloc[:,7].fillna(str(Dataset.iloc[:,7].mode()[0]))


#checking missing values
print(Dataset.iloc[:,:].isnull().sum())




#Breaking down into X and y

X = Dataset.iloc[:,1:].values
y = Dataset.iloc[:,0].values


#converting categorical to label encoder
from sklearn.preprocessing import LabelEncoder
label_encoder_X1 = LabelEncoder()
X[:,0] = label_encoder_X1.fit_transform(X[:,0])
label_encoder_X2 = LabelEncoder()
X[:,1] = label_encoder_X2.fit_transform(X[:,1])
label_encoder_X3 = LabelEncoder()
X[:,6] = label_encoder_X3.fit_transform(X[:,6])
label_encoder_X4 = LabelEncoder()





#applyying onehot encoder
from sklearn.preprocessing import OneHotEncoder
OHE = OneHotEncoder(categorical_features = [0,1,6])
X = OHE.fit_transform(X).toarray()



#dividing training and Test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)




#applying Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)



##fitting SVM rbf kernel classifier
#from sklearn.svm import SVC
#classifier = SVC(kernel = 'rbf' , random_state= 0)
#classifier.fit(X_train,y_train)

##fitting  SVM poly classifier
#from sklearn.svm import SVC
#classifier = SVC(kernel = 'poly' ,degree= 2, random_state= 0)
#classifier.fit(X_train,y_train)


##fitting KNN
#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors=15,metric = 'minkowski',p=2)
#classifier.fit(X_train,y_train)

##fitting decision tree
#from sklearn.tree import DecisionTreeClassifier
#classifier = DecisionTreeClassifier(criterion = 'entropy', random_state =0)
#classifier.fit(X_train,y_train)

#fitting random forest classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train,y_train)

##fitting logistic regression
#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression(random_state = 0)
#classifier.fit(X_train,y_train)


#predicting y
y_pred = classifier.predict(X_test)
#y_pred1 = classifier.predict(X)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)






#applying on actual test data
Dataset_test = pd.read_csv('C:\\A_stuff\\Learning\\Machine Learning\\Kaggle\\Titanic Problem\\test.csv')


#checking head of dataset
print(Dataset_test.head(2))


#deleting unncessary data
Dataset_test =Dataset_test.drop(['PassengerId','Name','Ticket','Cabin'],axis =1)



#checking missing values
print(Dataset_test.iloc[:,:].isnull().sum())



#filling missing values
Dataset_test.iloc[:,2] = Dataset_test.iloc[:,2].fillna(str(Dataset_test.iloc[:,2].mean()))
Dataset_test.iloc[:,5] = Dataset_test.iloc[:,5].fillna(str(Dataset_test.iloc[:,5].mode()[0]))


#checking missing values
print(Dataset_test.iloc[:,:].isnull().sum())




#getting featues
X_actualtest = Dataset_test.iloc[:,:].values

#for new test data (label enconder and onehot encoder)
X_actualtest[:,0] = label_encoder_X1.transform(X_actualtest[:,0])
X_actualtest[:,1] = label_encoder_X2.transform(X_actualtest[:,1])
X_actualtest[:,6] = label_encoder_X3.transform(X_actualtest[:,6])



#applyying onehot encoder
X_actualtest = OHE.transform(X_actualtest).toarray()


#applying Feature scaling
X_actualtest = sc_X.transform(X_actualtest)


#predicting y
y_pred = classifier.predict(X_actualtest)
A = pd.DataFrame(y_pred)

from openpyxl import load_workbook
writer = pd.ExcelWriter('C:\\A_stuff\\Learning\\Machine Learning\\Kaggle\\Titanic Problem\\Output.xls')
A.to_excel(writer,'Sheet2')
writer.save()





