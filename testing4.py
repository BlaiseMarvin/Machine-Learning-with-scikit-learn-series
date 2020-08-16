from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

dataset=load_digits()

X,y=dataset['data'],dataset['target']==1

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

clf=SVC(kernel='rbf')
grid_values={'gamma':[0.001,0.01,0.05,0.1,1,10,100],'C':[0.001,0.1,1,10,100,1000]}

grid_clf_acc=GridSearchCV(clf,param_grid=grid_values)
grid_clf_acc.fit(X_train,y_train)

print("Grid best parameter which gives maximum accuracy: ",grid_clf_acc.best_params_)
print("Grid best score: ",grid_clf_acc.best_score_)

grid_clf_roc=GridSearchCV(clf,param_grid=grid_values,scoring='roc_auc')
grid_clf_roc.fit(X_train,y_train)

print("Best roc parameter ",grid_clf_roc.best_params_)
print("Best roc value ",grid_clf_roc.best_score_)