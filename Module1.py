#applied machine learning 
#a simple classification task
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

fruits=pd.read_table('fruit_data_with_colors.txt')

print(fruits.head())

#create a mapping from fruit label to fruit name to make results easier to interprete

lookup_fruit_name=dict(zip(fruits.fruit_label.unique(),fruits.fruit_name.unique()))
print(lookup_fruit_name)

#examining the data

X=fruits[['mass','width','height','color_score']]
y=fruits['fruit_label']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

#scatter matrix below examining the data
from pandas.plotting import scatter_matrix

#scatter_matrix(X_train,diagonal='kde',c=y_train,marker='o',alpha=0.5)

#plotting a 3D scatter plot
from mpl_toolkits.mplot3d import Axes3D

fig=plt.figure()

#ax=fig.add_subplot(1,1,1, projection='3d')
#ax.scatter(X_train['width'],X_train['height'],X_train['color_score'],c=y_train,marker='o',s=100)
#ax.set_xlabel('width')
#ax.set_ylabel('height')
#ax.set_zlabel('color_score')

#plt.show()

#Create a train_test_split

#now create a classifier object

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)

#now let's train the classifier using the training data
knn.fit(X_train,y_train)

#estimate the accuracy, using the test data
score=knn.score(X_test,y_test)
print(score)

#use the trained, knn classifier model to classify new previously unseen objects
fruit_prediction=knn.predict([[20,4.3,5.5,0.7]])
print(fruit_prediction)

print(lookup_fruit_name[fruit_prediction[0]])


fruit_prediction=knn.predict([[100,6.3,8.5,0.5]])
print(lookup_fruit_name[fruit_prediction[0]])

#let's plot the decision boundaries of the knn classifier

#how sensitive is the accuracy to the choice of k parameter

k_range=range(1,20)
scores=[]

for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    score=knn.score(X_test,y_test)
    scores.append(score)

#print(scores)
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
#ax.plot(k_range,scores,'ko-')
ax.set_xlabel('range of k values')
ax.set_ylabel('accuracy percentages')
ax.set_xticks([0,5,10,15,20])
#plt.show()

#how sensitive is the knn classification to the train/test split proportion

t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

knn=KNeighborsClassifier(n_neighbors=5)
plt.figure()

for s in t:
    scores=[]
    for i in range(1,1000):
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1-s)
        knn.fit(X_train,y_train)
        scores.append(knn.score(X_test,y_test))

    plt.plot(s,np.mean(scores),'bo')
plt.xlabel('Training set proportion (%)')
plt.ylabel('accuracy')
plt.show()