#Regression Eval
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LinearRegression

from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error,r2_score,median_absolute_error

diabetes= datasets.load_diabetes()

y=diabetes['target']
X=diabetes.data[:,None,6]
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

lr=LinearRegression().fit(X_train,y_train)
dr=DummyRegressor(strategy='mean').fit(X_train,y_train)

lr_pred=lr.predict(X_test)
dr_pred=dr.predict(X_test)

print("lr r2 score: {:.2f} ".format(r2_score(y_test,lr_pred)))
print("dr r2 score: {:.2f}  ".format(r2_score(y_test,dr_pred)))



