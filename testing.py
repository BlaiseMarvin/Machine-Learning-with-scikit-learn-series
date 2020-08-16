import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import PolynomialFeatures

poly=PolynomialFeatures(degree=1)

dataset=load_digits()
X,y=dataset['data'],dataset['target']


X_trainmc,X_testmc,y_trainmc,y_testmc=train_test_split(X,y,random_state=0)
X_train_poly=poly.fit_transform(X_trainmc)
X_test_poly=poly.transform(X_testmc)

from sklearn.svm import SVC
svm=SVC(kernel='rbf',C=1).fit(X_train_poly,y_trainmc)
predo=svm.predict(X_test_poly)

from sklearn.metrics import confusion_matrix

confyu=confusion_matrix(y_testmc,predo)

sns.heatmap(confyu,annot=True)
plt.ylabel=("True label")
plt.xlabel("Model Prediction")
#plt.show()

from sklearn.metrics import accuracy_score,precision_score,recall_score

print("Macro average precision: {:.2f} ".format(precision_score(y_testmc,predo,average='macro')))
print("Micro average precision: {:.2f} ".format(precision_score(y_testmc,predo,average='micro')))

from sklearn.metrics import classification_report

print(classification_report(y_testmc,predo,target_names=['0','1','2','3','4','5','6','7','8','9']))

