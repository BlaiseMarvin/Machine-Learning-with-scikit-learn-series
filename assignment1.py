import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

df=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df['target']=pd.DataFrame(cancer['target'])

Target=df.target.value_counts(ascending=True)

Target.index="malignant benign".split()

X=df.iloc[:,0:30]
y=df.iloc[:,30:31]

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)



