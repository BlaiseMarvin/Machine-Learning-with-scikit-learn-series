from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
import numpy as np 

cancer=load_breast_cancer()

knn=KNeighborsClassifier(n_neighbors=5)
scores=cross_val_score(knn,cancer['data'],cancer['target'],cv=3)

print(scores)
print("Mean score: {} ".format(np.mean(scores)))