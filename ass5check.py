import numpy as np 

a=np.array([[1,2,3,4],[5,6,7,8]])

print(a.mean(axis=1).shape)
print(a.mean(axis=0).shape)