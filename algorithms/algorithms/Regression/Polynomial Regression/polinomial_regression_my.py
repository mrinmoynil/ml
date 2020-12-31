#import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#import the data set
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:, 1:2].values # sh udhu 1 dile vector hoy.amdr drkr matrix
y=dataset.iloc[:, 2].values




#fitting  linear regression to dataset
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)


#fitting polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)

poly_reg.fit(X_poly,y)

lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,y)


#visualising the linear regression result
plt.scatter(X,y,color='red')
plt.scatter(X,lin_reg.predict(X),color='blue')
plt.title('truth or bluf(linear regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()


#visualising the polynomial regression result
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1) #CURVE SMOOTH KORE,MORE CONTINUOUS

plt.scatter(X,y,color='red')
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('truth or bluf(polynomial regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

a=np.array([6.5])
new_a=a.reshape(-1,1)
lin_reg.predict(new_a)

lin_reg_2.predict(poly_reg.fit_transform(new_a))
