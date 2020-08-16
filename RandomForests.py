from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X_train,X_test,y_train,y_test=train_test_split(cancer['data'],cancer['target'],random_state=0)

clf=RandomForestClassifier(max_features=8,random_state=0)
clf.fit(X_train,y_train)

print("Breast Cancer dataset")
print("Accuracy of RF classifier on the training set {:.2f} ".format(clf.score(X_train,y_train)))
print("Accuracy of RF classifier on the test set: {:.2f} ".format(clf.score(X_test,y_test)))

