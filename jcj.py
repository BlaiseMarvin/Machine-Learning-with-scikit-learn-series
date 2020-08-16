#Decision Trees
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
import numpy as np 
from IPython.display import Image
import pydotplus


iris=load_iris()

X_train,X_test,y_train,y_test=train_test_split(iris['data'],iris['target'],random_state=3)

clf=DecisionTreeClassifier().fit(X_train,y_train)

print("Accuracy of Decision tree classifier on training set: {:.3f}".format(clf.score(X_train,y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))


#from above it's clear that this model is overfitting

#setting max decision tree depth to avoid overfitting
from adspy_shared_utilities import plot_decision_tree
import graphviz
from sklearn.tree import export_graphviz
import io

clf2=DecisionTreeClassifier(max_depth=3).fit(X_train,y_train)

print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf2.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf2.score(X_test, y_test)))

def plot_decision_tree(clf2,feature_names,class_names):
    dot_data=io.StringIO()
    export_graphviz(clf,out_file=dot_data,feature_names=feature_names,class_names=class_names,filled=True,impurity=False,special_characters=True)
    graph=pydotplus.graphviz.graph_from_dot_data(dot_data.getvalue())
    return Image(graph.create_png())

plot_decision_tree(clf2, iris['feature_names'], iris['target_names'])
