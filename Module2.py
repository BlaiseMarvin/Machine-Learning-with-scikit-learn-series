import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split

#supervised learning part 2
#preamble and review

fruits=pd.read_table('fruit_data_with_colors.txt')
print(fruits.head())

feature_names_fruits=['mass','width','height','color_score']
X_fruits=fruits[feature_names_fruits]

y_fruits=fruits['fruit_label']

target_names=fruits['fruit_name'].unique()

X_fruits_2d=fruits[['height','width']]
y_fruits_2d=fruits['fruit_label']

X_train,X_test,y_train,y_test=train_test_split(X_fruits,y_fruits,random_state=0)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

X_train_scaled=scaler.fit_transform(X_train)

#applying the scaling to the test set as well
X_test_scaled=scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_scaled,y_train)
print("Accuracy on the training set: {:.2f}".format(knn.score(X_train_scaled,y_train)))
print("Accuracy on the test set : {:.2f} ".format(knn.score(X_test_scaled,y_test)))

example_fruit=np.array([[5.5,2.2,10,0.70]])
ex_scaled=scaler.transform(example_fruit)

predictions=dict(zip(fruits['fruit_label'].unique(),fruits['fruit_name'].unique()))

pred=knn.predict(ex_scaled)
print(pred)

#data sets
from sklearn.datasets import make_classification, make_blobs
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer
from adspy_shared_utilities import load_crime_dataset

cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])

#synthetic dataset for simple regression

from sklearn.datasets import make_regression

#fig=plt.figure()
#ax=fig.add_subplot(1,1,1)
#ax.set_title('Sample regression problem with one input variable')
X_R1, y_R1 = make_regression(n_samples = 100, n_features=1,n_informative=1, bias = 150.0,noise = 30, random_state=0)

#ax.scatter(X_R1,y_R1,marker='o',s=50)

#plt.show()

#Synthetic dataset for classification (binary)

#with classes that aren't linearly separable
X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2, centers = 8,cluster_std = 1.3, random_state = 4)
y_D2 = y_D2 % 2
#plt.figure()
plt.title('Sample binary classification problem with non-linearly separable classes')
#plt.scatter(X_D2[:,0], X_D2[:,1], c=y_D2,
           #marker= 'o', s=50, cmap=cmap_bold)
#plt.show()

# synthetic dataset for more complex regression
from sklearn.datasets import make_friedman1
#plt.figure()
#plt.title('Complex regression problem with one input variable')
X_F1, y_F1 = make_friedman1(n_samples = 100,
                          n_features = 7, random_state=0)

#plt.scatter(X_F1[:, 2], y_F1, marker= 'o', s=50)
#plt.show()

# synthetic dataset for classification (binary) 
#plt.figure()
#plt.title('Sample binary classification problem with two informative features')
X_C2, y_C2 = make_classification(n_samples = 100, n_features=2,n_redundant=0, n_informative=2,n_clusters_per_class=1, flip_y = 0.1,class_sep = 0.5, random_state=0)
#plt.scatter(X_C2[:, 0], X_C2[:, 1], c=y_C2,marker= 'o', s=50, cmap=cmap_bold)
#plt.show()

#Linear models for regression

#Linear regression (which uses the least squares method to derive the linear model that is used for approximating the data)

from sklearn.linear_model import LinearRegression

X_train,X_test,y_train,y_test=train_test_split(X_R1,y_R1,random_state=0)

linreg=LinearRegression().fit(X_train,y_train)

print("Linear model coeff (w): {} ".format(linreg.coef_))
print("Linear model intercept(b): {} ".format(linreg.intercept_))
print("R-squared score (training):{:.3f} ".format(linreg.score(X_train,y_train)))
print("R-squared score (test): {:.3f} ".format(linreg.score(X_test,y_test)))

#visualization of what actually has happened
#plt.figure(figsize=(5,4))
#plt.scatter(X_R1,y_R1,marker='o',s=50,alpha=0.8)
#plt.plot(X_R1,linreg.coef_*X_R1+linreg.intercept_,'r-')
#plt.title("Least squares linear regression")
#plt.xlabel("Feature value (x)")
#plt.ylabel("Target value (y) ")
#plt.show()


(X_crime,y_crime)=load_crime_dataset()
#Ridge regression

from sklearn.linear_model import Ridge

X_train,X_test,y_train,y_test=train_test_split(X_crime,y_crime,random_state=0)

linridge=Ridge(alpha=20.0).fit(X_train,y_train)
print('Crime dataset')
print('ridge regression linear model intercept: {}'
     .format(linridge.intercept_))
print('ridge regression linear model coeff:\n{}'
     .format(linridge.coef_))
print('R-squared score (training): {:.3f}'
     .format(linridge.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'
     .format(linridge.score(X_test, y_test)))
print('Number of non-zero features: {}'
     .format(np.sum(linridge.coef_ != 0)))

#Ridge regression with feature normalization

#It is a good practice to normalize features, so that the weighting is applied equally

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

linrig=Ridge(alpha=20.0).fit(X_train_scaled,y_train)

print("Now working with scaled data")
print("Crime dataset")
print("Ridge regression linear model intercept:{} ".format(linrig.intercept_))
print("Ridge regression linear model coefficient: {}".format(linrig.coef_))
print("R-squared score (training): {:.3f} ".format(linrig.score(X_train_scaled,y_train)))
print("R-squared score (test): {:.3f} ".format(linrig.score(X_test_scaled,y_test)))

print("Number of non zero features: {}".format(np.sum(linrig.coef_!=0)))


#ridge regression with regularization parameter: alpha

print("Ridge regression, effect of alpha regularization parameter\n")

for this_alpha in [0,1,10,20,50,100,1000]:
     linridge=Ridge(alpha=this_alpha).fit(X_train_scaled,y_train)
     r2_train=linridge.score(X_train_scaled,y_train)
     r2_test=linridge.score(X_test_scaled,y_test)
     num_coeff_bigger=np.sum(abs(linridge.coef_)>1.0)
     print('Alpha={:.2f}\n num abs(coeff)>1.0:{},\ r-squared training: {:.2f}, r-squared test: {:.2f}\n'.format(this_alpha, num_coeff_bigger, r2_train, r2_test)) 


from sklearn.linear_model import Lasso

linlasso=Lasso(alpha=2.0,max_iter=10000).fit(X_train_scaled,y_train)
print("Lasso regression:")
print("Crime data")
print("Lasso regression linear model intercept: {} ".format(linlasso.intercept_))
print("Lasso regression linear model coeff:\n{} ".format(linlasso.coef_))
print("Non zero features: {} ".format(np.sum(linlasso.coef_!=0)))

print("R-squared score (training): {:.3f} ".format(linlasso.score(X_train_scaled,y_train)))
print("R-squared score (test):{:.3f} ".format(linlasso.score(X_test_scaled,y_test)))

for e in sorted(list(zip(list(X_crime),linlasso.coef_)),key=lambda e: -abs(e[1])):

     if e[1] !=0:
          print('\t{},{:.3f}'.format(e[0],e[1]))

#Lasso regression with regularized parameter

print('Lasso regression: effect of alpha regularization\n\parameter on a number of features kept in final model \n ')

for alpha in [0.5,1,2,3,5,10,20,50]:
     linlasso=Lasso(alpha,max_iter=10000).fit(X_train_scaled,y_train)
     r2_train=linlasso.score(X_train_scaled,y_train)
     r2_test=linlasso.score(X_test_scaled,y_test)

     print('Alpha = {:.2f}\nFeatures kept: {}, r-squared training: {:.2f}, \
r-squared test: {:.2f}\n'
         .format(alpha, np.sum(linlasso.coef_ != 0), r2_train, r2_test))


#Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2)
X_F1_poly=poly.fit_transform(X_F1)

X_train,X_test,y_train,y_test=train_test_split(X_F1_poly,y_F1,random_state=0)

linreg=LinearRegression().fit(X_train,y_train)



print('(poly deg 2) linear model coeff (w):\n{}'.format(linreg.coef_))

print('(poly deg 2) linear model intercept (b):{:.3f} '.format(linreg.intercept_))
print('(poly deg 2) R-squared score (training): {:.3f} '.format(linreg.score(X_train,y_train)))

print('(poly deg 2) R-squared score (test): {:.3f}'.format(linreg.score(X_test,y_test)))


#addition of many polynomial features often leads to overfitting, henceforth polynomial features are employed with an algorithm that's got a regularization parameter

linreg=Ridge().fit(X_train,y_train)
print('(poly deg 2 + ridge) linear model coeff (w):\n{}'
     .format(linreg.coef_))
print('(poly deg 2 + ridge) linear model intercept (b): {:.3f}'
     .format(linreg.intercept_))
print('(poly deg 2 + ridge) R-squared score (training): {:.3f}'
     .format(linreg.score(X_train, y_train)))
print('(poly deg 2 + ridge) R-squared score (test): {:.3f}'
     .format(linreg.score(X_test, y_test)))


#Logistic Regression

from sklearn.linear_model import LogisticRegression

from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()

X_train,X_test,y_train,y_test=train_test_split(cancer['data'],cancer['target'],random_state=0)

clf=LogisticRegression(C=100).fit(X_train,y_train)
print("Breast cancer dataset")

print("Accuracy of logistic regression classifier on training set: {:.3f} ".format(clf.score(X_train,y_train)))

print("Accuracy on test set: {:.3f}".format(clf.score(X_test,y_test)))


from sklearn.svm import SVC
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot


X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state = 0)

fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))
this_C = 1.0
clf = SVC(kernel = 'linear', C=this_C).fit(X_train, y_train)
title = 'Linear SVC, C = {:.3f}'.format(this_C)
plot_class_regions_for_classifier_subplot(clf, X_train, y_train, None, None, title, subaxes)
#plt.show()

#Linear support vector machine C parameter

from sklearn.svm import LinearSVC
from adspy_shared_utilities import plot_class_regions_for_classifier

X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state = 0)
fig, subaxes = plt.subplots(1, 2, figsize=(8, 4))

for this_C, subplot in zip([0.00001, 100], subaxes):
    clf = LinearSVC(C=this_C).fit(X_train, y_train)
    title = 'Linear SVC, C = {:.5f}'.format(this_C)
    plot_class_regions_for_classifier_subplot(clf, X_train, y_train,
                                             None, None, title, subplot)
plt.tight_layout()

#plt.show()

#Application to a real world dataset

X_train,X_test,y_train,y_test=train_test_split(cancer['data'],cancer['target'],random_state=0)

clf=LinearSVC().fit(X_train,y_train)

print("Breast cancer dataset")
print('Accuracy of Linear SVC classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Linear SVC classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))


#Multi class classification with linear models

#LinearSVC with M classes generates M one vs rest classifiers


#KERNELIZED SUPPORT VECTOR MACHINES

#application of SVMs to a real dataset: unnormalized data

X_train,X_test,y_train,y_test=train_test_split(cancer['data'],cancer['target'],random_state=0)

clf=SVC(C=10,kernel='rbf').fit(X_train,y_train)

print("Breast cancer dataset(unnormalized features)")
print("Accuracy on training set: {:.3f} ".format(clf.score(X_train,y_train)))
print("Accuracy on test set: {:.3f} ".format(clf.score(X_test,y_test)))

#Application to a normalized dataset

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

clf=SVC(kernel='rbf',gamma=1,C=30).fit(X_train_scaled,y_train)
print("Breast cancer dataset (normalized with MinMax scaling)")

print("RBF-kernel SVC training set accuracy: {:.3f} ".format(clf.score(X_train_scaled,y_train)))
print("Test set accuracy: {:.3f} ".format(clf.score(X_test_scaled,y_test)))


#Using cross validation to evaluate models

#an example based on a k-nn classifier with fruit dataset (2 features)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

clf=KNeighborsClassifier(n_neighbors=5)
X=cancer['data']
y=cancer['target']

cv_scores=cross_val_score(clf,X,y)

print("Cross validation scores(3-fold): ",cv_scores)
print("Mean cross validation score (3-fold) :{:.3f} ".format(cv_scores.mean()))


#Validation curve

from sklearn.svm import SVC
from sklearn.model_selection import validation_curve

param_range=np.logspace(-3,3,4)

train_scores,test_scores=validation_curve(SVC(),X,y,param_name='gamma',param_range=param_range,cv=3)

print(train_scores)
print(test_scores)


#Decision Trees

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

iris=load_iris()

X_train,X_test,y_train,y_test=train_test_split(iris['data'],iris['target'],random_state=3)

clf=DecisionTreeClassifier().fit(X_train,y_train)

print("Accuracy of Decision tree classifier on training set: {:.3f}".format(clf.score(X_train,y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))


#from above it's clear that this model is overfitting

#setting max decision tree depth to avoid overfitting
from adspy_shared_utilities import plot_decision_tree

clf2=DecisionTreeClassifier(max_depth=3).fit(X_train,y_train)

print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf2.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf2.score(X_test, y_test)))

plot_decision_tree(clf, iris.feature_names, iris.target_names)
