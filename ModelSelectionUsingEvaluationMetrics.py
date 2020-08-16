from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.datasets import load_digits

dataset=load_digits()
X,y=dataset['data'],dataset['target']==1

clf=SVC(kernel='linear',C=1)

print("Cross validation (accuracy) ",cross_val_score(clf,X,y,cv=5))
print("Cross validation (AUC)",cross_val_score(clf,X,y,scoring='roc_auc'))
print("Cross validation (recall) ",cross_val_score(clf,X,y,scoring='recall'))
