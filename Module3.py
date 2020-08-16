#evaluation for classification
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

dataset=load_digits()

X,y=dataset['data'],dataset['target']

for class_name,class_count in zip(dataset['target_names'],np.bincount(dataset['target'])):
    print(class_name,class_count)

#from above, all classes have ideally the same number of targets

#creating a dataset with imbalanced binary classes

y_binary_imbalanced=y.copy()
y_binary_imbalanced[y_binary_imbalanced!=1]=0

print("Original labels:\t",y[1:30])
print("New binary labels:\t",y_binary_imbalanced[1:30])

print(np.bincount(y_binary_imbalanced))

X_train,X_test,y_train,y_test=train_test_split(X,y_binary_imbalanced,random_state=0)

#accuracy of SVM classifier

from sklearn.svm import SVC

svm=SVC(kernel='rbf',C=1).fit(X_train,y_train)
print(svm.score(X_test,y_test))


#into dummy classifiers

from sklearn.dummy import DummyClassifier

#negative class (0) is most frequent

dummy_majority=DummyClassifier(strategy='most_frequent').fit(X_train,y_train)

y_predictions_dummy=dummy_majority.predict(X_test)

print(y_predictions_dummy)

score=dummy_majority.score(X_test,y_test)
print(score)

#if your accuracy is the same as that of the dummy classifier, it signifies that something wrong is with your model, or data

#you might consider change, such as the introduction of a different flavour, such as using a linear model instead of rbf

svm=SVC(kernel='linear',C=1).fit(X_train,y_train)

score=svm.score(X_test,y_test)
print(score)


#Confusion Matrices

from sklearn.metrics import confusion_matrix

#negative class is the most frequent remember

dummy_majority=DummyClassifier(strategy='most_frequent').fit(X_train,y_train)
y_majority_predicted=dummy_majority.predict(X_test)

confusion=confusion_matrix(y_test,y_majority_predicted)

print("Most frequent class(dummy_classifier)\n",confusion)

#produces random predictions with same class proportion as training set

dummy_classprop=DummyClassifier(strategy='stratified').fit(X_train,y_train)

y_classprop_predicted=dummy_classprop.predict(X_test)
confusion=confusion_matrix(y_test,y_classprop_predicted)
print(confusion)

svm=SVC(kernel='linear',C=1).fit(X_train,y_train)

svm_predicted=svm.predict(X_test)
confusion=confusion_matrix(y_test,svm_predicted)

print(confusion)

from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression().fit(X_train,y_train)
logreg_pred=logreg.predict(X_test)
confusion=confusion_matrix(y_test,logreg_pred)
print(confusion)

from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(max_depth=2).fit(X_train,y_train)
dt_pred=dt.predict(X_test)
confusion=confusion_matrix(y_test,dt_pred)
print(confusion)


#evaluation metrics for binary classification

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
print("Accuracy: {:.2f} ".format(accuracy_score(y_test,dt_pred)))
print("Precision: {:.2f} ".format(precision_score(y_test,dt_pred)))
print("Recall: {:.2f} ".format(recall_score(y_test,dt_pred)))
print("f1: {:.2f}".format(f1_score(y_test,dt_pred)))

#a combined report with all the above metrics

from sklearn.metrics import classification_report

print(classification_report(y_test,dt_pred,target_names=['not 1','1']))

print("Random class-proportional (dummy)\n",classification_report(y_test,y_classprop_predicted,target_names=['not 1','1']))
print("SVM\n",classification_report(y_test,svm_predicted,target_names=['not 1','1']))
print("Logistic regression\n",classification_report(y_test,logreg_pred,target_names=['not 1','1']))
print("Decision tree\n",classification_report(y_test,dt_pred,target_names=['not 1','1']))


#Decision functions 
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
X_train,X_test,y_train,y_test=train_test_split(X,y_binary_imbalanced,random_state=0)
y_scores_lr=lr.fit(X_train,y_train).decision_function(X_test)
y_score_list=list(zip(y_test[0:20],y_scores_lr[0:20]))

for z in y_score_list:
    print(z)

#Using predictproba to predict probabilities

X_train,X_test,y_train,y_test=train_test_split(X,y_binary_imbalanced,random_state=0)
lri=LogisticRegression()
y_proba_lr=lri.fit(X_train,y_train).predict_proba(X_test)
y_proba_list=list(zip(y_test[0:20],y_proba_lr[0:20,1]))

for zi in y_proba_list:
    print(zi)

#precision recall curves
#auc curves

#evaluation measures for multiclass classification

from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import pandas as pd 
import seaborn as sns
from sklearn.metrics import accuracy_score
dataset=load_digits()

X,y=dataset['data'],dataset['target']

X_train_mc,X_test_mc,y_train_mc,y_test_mc=train_test_split(X,y,random_state=0)

svm=SVC(kernel='linear').fit(X_train_mc,y_train_mc)
svm_predicted_mc=svm.predict(X_test_mc)
confusion_mc=confusion_matrix(y_test_mc,svm_predicted_mc)

df_cm=pd.DataFrame(confusion_mc,index=[i for i in range(0,10)],columns=[i for i in range(0,10)])

plt.figure(figsize=(5.5,4))
sns.heatmap(df_cm,annot=True)

plt.title("SVM Linear Kernel \n Accuracy:{0:.3f} ".format(accuracy_score(y_test_mc,svm_predicted_mc)))
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
#plt.show()

svm=SVC(kernel='rbf').fit(X_train_mc,y_train_mc)
svm_predicted_mc=svm.predict(X_test_mc)
confusion_mc=confusion_matrix(y_test_mc,svm_predicted_mc)
df_cm=pd.DataFrame(confusion_mc,index=[i for i in range(0,10)],columns=[i for i in range(0,10)])

plt.figure(figsize=(5.5,4))
sns.heatmap(df_cm,annot=True)
plt.title("SVM RBF Kernel \n Accuracy:{0:.3f} ".format(accuracy_score(y_test_mc,svm_predicted_mc)))
plt.xlabel('Predicted label')
plt.ylabel('True label')

#plt.show()

#Multiclass classification report

from sklearn.metrics import classification_report
from sklearn.metrics import precision_score,f1_score

print(classification_report(y_test_mc,svm_predicted_mc))

#Micro vs macro-averaged metrics
print('Micro-averaged precision = {:.2f} (treat instances equally)'.format(precision_score(y_test_mc,svm_predicted_mc,average='micro')))

print("Macro-averaged precision ={:.2f} (treat )")


