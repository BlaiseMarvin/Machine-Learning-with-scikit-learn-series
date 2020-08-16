import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.dummy import DummyRegressor

diabetes= datasets.load_diabetes()

y=diabetes['target']
X=diabetes.data[:,None,6]

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

lm=LinearRegression().fit(X_train,y_train)
lm_dummy_mean=DummyRegressor(strategy='mean').fit(X_train,y_train)

y_predict=lm.predict(X_test)
y_predict_dummy_mean=lm_dummy_mean.predict(X_test)

print("Linear model, coefficients: ",lm.coef_)
print("Mean squared error (dummy): {:.2f} ".format(mean_squared_error(y_test,y_predict_dummy_mean)))
print("Mean squared error (Linear Model): {:.2f} ".format(mean_squared_error(y_test,y_predict)))

print("r2 score (dummy): {:.2f} ".format(r2_score(y_test,y_predict_dummy_mean)))
print("r2 score (linear model): {:.2f} ".format(r2_score(y_test,y_predict)))

#Lets plot the outputs

plt.scatter(X_test,y_test,color='black')
plt.plot(X_test,y_predict,color='green',linewidth=2)
plt.plot(X_test,y_predict_dummy_mean,color='red',linewidth=2,linestyle='dashed',label='dummy')

plt.show()