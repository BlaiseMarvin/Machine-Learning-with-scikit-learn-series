#exploring the relationship between model complexity and generalization performance, by adjusting key parameters of various supervised learning models.
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(0)

n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

plt.scatter(X_train,y_train,label='Training data')
plt.scatter(X_test,y_test,label='Test data')
plt.legend(loc='best')

#plt.show()

#Question one

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
degrees =[1,3,6,9]
predictions=np.zeros((4,100))
"""for degree in degrees:
    linreg=LinearRegression()
    poly=PolynomialFeatures(degree=degree)
    X_train_poly=poly.fit_transform(X_train.reshape(11,1))
    linreg.fit(X_train_poly,y_train)
    pred_data=poly.transform(np.linspace(0,10,100).reshape(100,1))
    output=linreg.predict(pred_data)
    predictions[0:]=output
"""
#print(predictions)

#write a function that fits a polynomial Linear regression model on training data X_train for degrees 0 thru 9

traini=np.zeros((10,))
testi=np.zeros((10,))

for i,d in enumerate(range(0,10)):
    poly= PolynomialFeatures(degree=d)
    X_poly=poly.fit_transform(X_train.reshape(11,1))
    linreg=LinearRegression().fit(X_poly,y_train)
    traini[i:]=linreg.score(X_poly,y_train)

    X_test_poly=poly.fit_transform(X_test.reshape(4,1))
    testi[i:]=linreg.score(X_test_poly,y_test)

print(traini)
print(testi)

print((traini,testi))

import pandas as pd 

data={'training':traini,'testing':testi}
df=pd.DataFrame(data)
df['difference']=df['training']-df['testing']
print(df)

#which model is overfitting
dfo=df['difference']
dfoo=dfo.copy()
dfoo.sort_values(ascending=False,inplace=True)
overfitting=dfoo.index[0]

#which model is a good generalization
dfu=df['difference']
dfuu=dfu.copy()
dfuu.sort_values(ascending=True,inplace=True)
good_generalization=dfuu.index[0]

#which model is underfitting
dfg=df['training']
dfgg=dfg.copy()
dfgg.sort_values(ascending=True,inplace=True)
underfitting=dfgg.index[0]

ze_tup=(overfitting,underfitting,good_generalization)
print(ze_tup)


#Training two models.
from sklearn.linear_model import Lasso

linireg=LinearRegression()
poly=PolynomialFeatures(degree=12)
X_poly=poly.fit_transform(X_train.reshape(11,1))
X_test_poly=poly.fit_transform(X_test.reshape(4,1))

linireg.fit(X_poly,y_train)

lass=Lasso(alpha=0.01,max_iter=10000).fit(X_poly,y_train)

lintestscore=linireg.score(X_test_poly,y_test)
lasstestscore=lass.score(X_test_poly,y_test)

za_tup=(lintestscore,lasstestscore)
print(za_tup)


#Part 2: Classification

mush_df=pd.read_csv('mushrooms.csv')
mush_df2=pd.get_dummies(mush_df)

X_mush=mush_df2.iloc[:,2:]
y_mush=mush_df2.iloc[:,:1]

X_train2,X_test2,y_train2,y_test2=train_test_split(X_mush,y_mush,random_state=0)
X_subset = X_test2
y_subset = y_test2

from sklearn.tree import DecisionTreeClassifier

de_tree=DecisionTreeClassifier()
de_tree.fit(X_train2,y_train2)
feature_names=[]

for index,importance in enumerate(de_tree.feature_importances_):
    feature_names.append([importance,mush_df2.columns[index]])

feature_names.sort(reverse=True)
feature_name=np.array(feature_names)
print(feature_name)

#top 5 features
name=feature_name[:5,1]
list_name=name.tolist()
print(list_name)


#Qn 6
#using validation curve
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve

gemme=np.logspace(-4,1,6)

training_scores,test_scores=validation_curve(SVC(),X_subset,y_subset,param_name='gamma',param_range=gemme,scoring='accuracy')

ze_training_scores=training_scores.mean(axis=1)
ze_test_scores=test_scores.mean(axis=1)

print(ze_training_scores)
print(ze_test_scores)

the_scores=(ze_training_scores,ze_test_scores)

print(the_scores)

#what gamma value corresponds to overfitting,underfitting and good_generalization

ze_final={'gamma':gemme,'training':ze_training_scores,'testing':ze_test_scores}
df=pd.DataFrame(ze_final)
print(df)

#0verfitting
df.set_index('gamma',inplace=True)
df['difference']=df['training']-df['testing']
dfo=df['difference']
dfoo=dfo.copy()

dfoo.sort_values(ascending=False,inplace=True)
overfit=dfoo.index[0]

#Underfitting
dfu=df['training']
dfuu=dfu.copy()

dfuu.sort_values(ascending=True,inplace=True)
underfit=dfuu.index[0]

#good generalization
dfg=dfo.copy()

dfg.sort_values(ascending=True,inplace=True)
good_gen=dfg.index[0]

za_tuple=(underfit,overfit,good_gen)

print(za_tuple)




