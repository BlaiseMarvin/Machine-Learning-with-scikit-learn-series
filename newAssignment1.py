#Introduction to machine learning

import numpy as np
import pandas as pd 

from sklearn.datasets import load_breast_cancer

cancer=load_breast_cancer()

print(len(cancer['feature_names']))

#converting the data to a dataframe

df=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df['target']=cancer['target']
print(df.head())
print(df.shape)

instances={}
z=df['target'].value_counts(ascending=True)
z.rename(index={0:'malignant',1:'beningn'},inplace=True)
print(z)

#split the dataframe into X the data and y the labels

X=df.iloc[:,:-1]
y=df.iloc[:,-1]

print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

#predict the class label using the mean value of each feature

means=df.mean()[:-1].values.reshape(1,-1)
print(means.shape)
prediction=knn.predict(means)
print(prediction)

#predict the class labels for the test set

predictions=knn.predict(X_test)

print(predictions)

score=knn.score(X_test,y_test)
print(score)

#Using the plotting function to visualize the different prediction scores between training and test set as well as malignant and benign cells
import matplotlib.pyplot as plt
def accuracy_plot():

    mal_train_X=X_train[y_train==0]
    mal_train_y=y_train[y_train==0]
    ben_train_X=X_train[y_train==1]
    ben_train_y=y_train[y_train==1]

    mal_test_X=X_test[y_test==0]
    mal_test_y=y_test[y_test==0]
    ben_test_X=X_test[y_test==1]
    ben_test_y=y_test[y_test==1]

    knn=KNeighborsClassifier(n_neighbors=1)

    knn.fit(X_train,y_train)
    scores=[knn.score(mal_train_X,mal_train_y),knn.score(ben_train_X,ben_train_y),knn.score(mal_test_X,mal_test_y),knn.score(ben_test_X,ben_test_y)]

    plt.figure()

    #plot the scores as bar charts

    bars=plt.bar(np.arange(4),scores,color=['#4c72b0','#4c72b0','#55a868','#55a868'])

    #directly label the scores onto the bars

    for bar in bars:
        height=bar.get_height()
        plt.gca().text(bar.get_x() +bar.get_width()/2,height)