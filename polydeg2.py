from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
import pandas as pd 
data=load_boston()
data_dataframe=pd.DataFrame(data['data'],columns=data['feature_names'])
Target=pd.DataFrame(data['target'])
data_dataframe['target']=Target

X=data_dataframe.iloc[:,0:13]
y=data_dataframe.iloc[:,13:14]

print("Now we transform the original input data to add \n Polynomial features up to degree 2 (quadratic) \n")
poly=PolynomialFeatures(degree=2)
X_poly=poly.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X_poly,y,random_state=0)

linreg=LinearRegression().fit(X_train,y_train)

print("Linear model coeff (w) {} ".format(linreg.coef_))
print("Linear model intercept (b) {} ".format(linreg.intercept_))
print("R-squared score (training) {:.3f} ".format(linreg.score(X_train,y_train)))
print("R-squared score (testing)  {:.3f} ".format(linreg.score(X_test,y_test)))