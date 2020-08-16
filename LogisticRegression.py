from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer=load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split(cancer['data'],cancer['target'],random_state=0)

clf=LogisticRegression().fit(X_train,y_train)

print("Breast Cancer dataset")
print("Accuracy of the logistic regression classifier on the training set:  {:.2f} ".format(clf.score(X_train,y_train)))
print("Accuracy of logistic regression on the test set:  {:.2f} ".format(clf.score(X_test,y_test)))
