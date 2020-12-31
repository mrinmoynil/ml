import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#import the data set
dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:, :-1].values   # .values deyar karone index dekhay nai,khali value dekhaiche
y=dataset.iloc[:,4].values


#encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])

onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()


#avoiding dummy variable trap
X=X[:,1:]   #taking coloum 1 to 5 except coloum 0 of X 


#splitting dataset into traing & testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


#fitting in training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)


#prediction
y_pred=regressor.predict(X_test)



 
 