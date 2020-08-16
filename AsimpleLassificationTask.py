import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D



fruits=pd.read_table('fruit_data_with_colors.txt')

X=fruits[['mass','width','height']]
y=fruits['fruit_label']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

#we need to visuzlise our training data, see if the problem can be solved with machine learning or not.
#below is how to visualize data in 2D

#cmap=cm.get_cmap('gnuplot')
#scatter=pd.plotting.scatter_matrix(X_train,c=y_train,marker='o',s=40,hist_kwds={'bins':15},figsize=(12,12),cmap=cmap)
#plt.show()

#Visualising data in 3D
ax=plt.axes(projection='3d')
ax.scatter3D(X_train['width'],X_train['height'],X_train['mass'],c=y_train,marker='o',s=100)



ax.set_xlabel('width')
ax.set_ylabel('height')
ax.set_zlabel('mass')

plt.show()
