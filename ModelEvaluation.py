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

#We now are going to create an imbalanced class
y_binary_imbalanced=y.copy()
y_binary_imbalanced[y_binary_imbalanced!=1]=0

print("Original labels:  ",y[1:30])
print("New binary labels: ",y_binary_imbalanced[1:30]) #this is now the imbalanced class' labels, most of the data as we can see corresponds to class zero, and very few instances actually correspond to class 1

#Let's now use the bincount method in numpy and visualise how many instances in our dataset belong to zero and how many belong to one
print(np.bincount(y_binary_imbalanced))

#Let's now train and fit a model on this inbalanced dataset
from sklearn.svm import SVC
X_train,X_test,y_train,y_test=train_test_split(X,y_binary_imbalanced,random_state=0)

svm = SVC(kernel='rbf',C=1).fit(X_train,y_train)
scores=svm.score(X_test,y_test)
print(scores)
#The accuarcy here, that is denoted by the scores value printed is so high. We might at first think, that our model is actually doing great. Great generalization accuracy

#Let's now create a dummy classifier

from sklearn.dummy import DummyClassifier
dummy_majority=DummyClassifier(strategy='most_frequent').fit(X_train,y_train)

#we create the dummy classifier, create an instance of it and then using the strategy method give it a strategy for making predictions
#this classifier won't even look at the X data, it will only look at the labels, and isolate out the most frequent label
#It will always assign each instance the most frequent label as told to it by the strategy parameter

y_dummy_predictions=dummy_majority.predict(X_test)

#try checking out the predictions for this X_test, all the predictions actually correspond to the majority class
print(y_dummy_predictions)

print(dummy_majority.score(X_test,y_test))
#as we can see the accuracy of this dummy class, that literally is a dummy is also 90%, in otherwords the accuracy is also so good.
#this basically shows us the limitations of this score method. The limitations of using accuracy to evaluate a model.

svm=SVC(kernel='linear',C=1).fit(X_train,y_train)
print(svm.score(X_test,y_test))


#Use of a confusion matrix
#If the task ahead is a binary predicition task, then the confusion matrix is a 282 matrix, and if the task ahead is a multiclassification problem, then the confusion matrix is a k*k matrix
#confusion matrix contains labels of true positive, true negatives, and also type 1 errors like false positive and also type 2 errors like false negatives

#binary(two-class) confusion matrix
from sklearn.metrics import confusion_matrix
dummy_majority=DummyClassifier(strategy='most_frequent').fit(X_train,y_train)

y_majority_predicted=dummy_majority.predict(X_test)
confusion = confusion_matrix(y_test,y_majority_predicted)

print("Most frequent class(dummy_classifier)\n",confusion)

#Let's spice things up a little and now try using the dummy classifier with the stratified strategy that produces random outputs in correspondence to the distribution in the training set
dummy_classprop=DummyClassifier(strategy='stratified').fit(X_train,y_train)
y_classprop_predicted=dummy_classprop.predict(X_test)
confusion=confusion_matrix(y_test,y_classprop_predicted)

print("Random class-proportional prediction(dummy classifier)\n",confusion)

#Using a support vector machine classifier
svm=SVC(kernel='poly',C=1).fit(X_train,y_train)
svm_predicted=svm.predict(X_test)
confusion=confusion_matrix(y_test,svm_predicted)
print("Support Vector Machine Classifier (kernel=linear,C=1)\n",confusion)

#using a logistic regression classifier
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression().fit(X_train,y_train)
lr_predicted=lr.predict(X_test)
confusion=confusion_matrix(y_test,lr_predicted)

print("Logistic Regression classifier(Default settings)\n",confusion)

#Let's now apply a decision tree classifier and look at the confusion matrix that results

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(max_depth=2).fit(X_train,y_train)
dt_prediction=dt.predict(X_test)
confusion=confusion_matrix(y_test,dt_prediction)
print("Decision Tree Classifier (max_depth=2) \n",confusion)

#We now want to compute our evaluation metrics, i.e. the accuracy, precision and recall

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

print("Accuracy: {:.2f} ".format(accuracy_score(y_test,dt_prediction)))
print("Precision: {:.2f} ".format(precision_score(y_test,dt_prediction)))
print("Recall:  {:.2f} ".format(recall_score(y_test,dt_prediction)))
print("F1:  {:2F} ".format(f1_score(y_test,dt_prediction)))

#using the classification report function

from sklearn.metrics import classification_report
print(classification_report(y_test,dt_prediction,target_names=['not 1','1']))
print(classification_report(y_test,lr_predicted,target_names=['not 1','1']))
print(classification_report(y_test,svm_predicted,target_names=['not 1','1']))
print(classification_report(y_test,y_classprop_predicted,target_names=['not 1','1']))


#DECISION FUNCTIONS
X_train,X_test,y_train,y_test=train_test_split(X,y_binary_imbalanced,random_state=0)

y_scores_lr=lr.fit(X_train,y_train).decision_function(X_test)
y_score_list=list(zip(y_test[0:20],y_scores_lr[0:20]))
print(y_score_list)


y_predictproba_scores=lr.fit(X_train,y_train).predict_proba(X_test)
y_predictproba_list=list(zip(y_test[0:20],y_predictproba_scores[0:20,1]))
print(y_predictproba_list)




