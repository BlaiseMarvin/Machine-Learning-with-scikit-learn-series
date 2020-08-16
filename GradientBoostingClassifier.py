from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer=load_breast_cancer()

X_train,X_test,y_train,y_test=train_test_split(cancer['data'],cancer['target'],random_state=0)

clf=GradientBoostingClassifier(random_state=0).fit(X_train,y_train)
#using default parameters such as the default learning_rate of 0.1

print("Breast cancer dataset (learning rate=0.1,max depth =3) ")
print("Accuracy on training set: {:.2f} ".format(clf.score(X_train,y_train)))
print("Accuracy on the test set: {:.2f} ".format(clf.score(X_test,y_test)))


clfd=GradientBoostingClassifier(learning_rate=0.01,max_depth=2,random_state=0).fit(X_train,y_train)
print("Breast cancer dataset (learning_rate = 0.01, max_depth=2) ")
print("Accuracy on training set: {:.2f} ".format(clfd.score(X_train,y_train)))
print("Accuracy on test set:  {:.2f} ".format(clfd.score(X_test,y_test)))

#here, initially the training accuracy was fixed at one, implying that there was significant overfitting. 
#to make the models less complex, we reduced the learning rate and also reduced the max_depth and as can be seen, the training accuracy reduced and the test accuracy increased
