
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)


result=np.zeros((4,100))
for i, degree in enumerate([1,3,6,9]):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_train.reshape(11,1))
    linreg = LinearRegression().fit(X_poly, y_train)
        
    y = linreg.predict(poly.fit_transform(np.linspace(0,10,100).reshape(100,1)));
    result[i,:] = y
    
print(result.shape)
print(result)