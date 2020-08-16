from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer=load_breast_cancer()
scaler=MinMaxScaler()
X_train,X_test,y_train,y_test=train_test_split(cancer['data'],cancer['target'],random_state=0)

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

clf=MLPClassifier(hidden_layer_sizes=[100,100],alpha=5.0,random_state=0,solver='lbfgs').fit(X_train_scaled,y_train)

print("Breast Cancer dataset")
print("Accuracy of NN classifier on the training set: {:.2f} ".format(clf.score(X_train_scaled,y_train)))
print("Accuracy of NN classifier on the test set: {:.2f} ".format(clf.score(X_test_scaled,y_test)))