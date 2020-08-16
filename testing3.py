from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_digits
from sklearn.svm import SVC


dataset=load_digits()
X,y=dataset['data'],dataset['target']

ynew=y.copy()
ynew[ynew!=1]=0

svm=SVC(kernel='rbf',gamma=10,C=1)


print("Accuracy: ",cross_val_score(svm,X,y,cv=5))

print("Recall: ",cross_val_score(svm,X,,cv=5,scoring='recall'))

