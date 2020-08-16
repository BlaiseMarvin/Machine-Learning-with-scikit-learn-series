from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
from sklearn.datasets import load_breast_cancer
import numpy as np 
cancer=load_breast_cancer()
param_range = np.logspace(-3,3,4)
train_scores,test_scores=validation_curve(SVC(),cancer['data'],cancer['target'],param_name='gamma',param_range=param_range,cv=3)
print(train_scores)
print(test_scores)

#this is what 