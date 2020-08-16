from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from adspy_shared_utilities import load_crime_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import mglearn
import numpy as np 


crime = pd.read_table('CommViolPredUnnormalizedData.txt', sep=',', na_values='?')
columns_to_keep = [5, 6] + list(range(11,26)) + list(range(32, 103)) + [145]  
crime = crime.iloc[:,columns_to_keep].dropna()

X_crime = crime.iloc[:,range(0,88)]
y_crime = crime['ViolentCrimesPerPop']



scaler = MinMaxScaler()
X_train,X_test,y_train,y_test= train_test_split(X_crime,y_crime,random_state=0)

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

linlasso = Lasso(alpha=2.0,max_iter=10000).fit(X_train_scaled,y_train)

print("Crime dataset")
print("Lasso regression linear model intercept:\n {}".format(linlasso.intercept_))
print("Lasso regression linear model coefficient:\n {} ".format(linlasso.coef_))
print("Non Zero features: {}".format(np.sum(linlasso.coef_!=0)))
print("R-squared score (training) {:.3f} ".format(linlasso.score(X_train_scaled,y_train)))
print("R-squared score (test) {:.3f} ".format(linlasso.score(X_test_scaled,y_test)))

print("Features with non zero weight, sorted by absolute magnitude ")



for e in sorted(list(zip(list(X_crime),linlasso.coef_)),key= lambda e: -abs(e[1])):
    if e[1]!=0:
        print('\t{},{:.3f}'.format(e[0],e[1]))


