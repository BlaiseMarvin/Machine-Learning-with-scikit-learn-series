

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    
    r2_train=np.zeros((10,))
    r2_test=np.zeros((10,))

    for x in range(0,10):
        poly=PolynomialFeatures(degree=x)
        X_poly=poly.fit_transform(X_train.reshape(11,1))
        linreg=LinearRegression()
        linreg.fit(X_poly,y_train)
        X_test_poly=poly.fit_transform(X_test.reshape(4,1))
        r2_train[x]=linreg.score(X_poly,y_train)
        r2_test[x]=linreg.score(X_test_poly,y_test)
        

    return (r2_train,r2_test)

r2_score=answer_two()
df=pd.DataFrame({'training_scores':r2_score[0],'test_scores':r2_score[1]})
df['Difference']=df['training_scores']-df['test_scores']

df=df.sort_values(['Difference'])

good_gen=df.index[0]

df=df.sort_values(['Difference'],ascending=False)
overfitting=df.index[0]

df=df.sort_values(['training_scores'])
underfitting=df.index[0]

