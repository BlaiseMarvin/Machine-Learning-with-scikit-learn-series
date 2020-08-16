import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10

print(x)
print(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    
    
    results=np.zeros((4,100))
    degrees=np.array([1,3,6,9])
    for i,d in enumerate(degrees):
        poly=PolynomialFeatures(degree=d)
        X_poly=poly.fit_transform(X_train.reshape(11,1))
        linreg=LinearRegression()
        linreg.fit(X_poly,y_train)
        predicting_data=poly.fit_transform(np.linspace(0,10,100).reshape(100,1))
        output=linreg.predict(predicting_data)
        results[i:]=output
        
    
    return results

def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics.regression import r2_score
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
def answer_three():
    r2_scores=answer_two()
    scores_dataframe=pd.DataFrame({'Training_Scores':r2_scores[0],'Test_Scores':r2_scores[1]})
    scores_dataframe['Difference']=scores_dataframe['Training_Scores']-scores_dataframe['Test_Scores']
    scores_dataframe=scores_dataframe.sort_values(['Difference'])
    Good_Generalization=scores_dataframe.index[0]
    
    scores_dataframe=scores_dataframe.sort_values(['Difference'],ascending=False)
    Overfitting=scores_dataframe.index[0]
    
    scores_dataframe=scores_dataframe.sort_values(['Training_Scores'])
    Underfitting=scores_dataframe.index[0]
    
    
    
    return (Underfitting,Overfitting,Good_Generalization)

print(answer_three())



