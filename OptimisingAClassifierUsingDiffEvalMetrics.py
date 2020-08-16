from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from adspy_shared_utilities import plot_class_regions_for_classifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np 
dataset=load_digits()

X,y=dataset['data'],dataset['target']==1
X_train,X_test,y_train,y_test= train_test_split(X,y,random_state=0)

jitter_delta=0.25
X_twovar_train=X_train[:,[20,59]] + np.random.rand(X_train.shape[0],2)-jitter_delta



