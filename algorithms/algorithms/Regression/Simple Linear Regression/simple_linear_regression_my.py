#simple linear regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#import the data set
dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:, :-1].values   # .values deyar karone index dekhay nai,khali value dekhaiche
y=dataset.iloc[:, 1].values


#splitting dataset into traing & testing
from sklearn.model_selection import train_test_split
 X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)
  

#fitting  imto training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)


#prediction
y_pred=regressor.predict(X_test)


#visualisation
plt.scatter(X_train,y_train,color="red")
plt.plot(X_train,regressor.predict(X_train))
plt.title('salary vs experience')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

plt.scatter(X_test,y_test,color="green")
plt.plot(X_train,regressor.predict(X_train))
plt.title('salary vs experience')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()