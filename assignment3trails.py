import pandas as pd 
import numpy as np

fraud_df=pd.read_csv('fraud_data.csv')
fraud_instances=len(fraud_df[fraud_df['Class']==1])
not_fraud=len(fraud_df[fraud_df['Class']==0])
print(not_fraud)
total=len(fraud_df['Class'])
print(total)

percentage=float(fraud_instances/total)
print(percentage)


from sklearn.model_selection import train_test_split

X=fraud_df.iloc[:,:-1]
y=fraud_df.iloc[:,-1]

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

from sklearn.dummy import DummyClassifier
from sklearn.metrics import recall_score

dc=DummyClassifier(strategy='most_frequent').fit(X_train,y_train)
accuracy_score=dc.score(X_test,y_test)
recall_score=recall_score(y_test,dc.predict(X_test))
print(accuracy_score)
print(recall_score)


from sklearn.svm import SVC
svm=SVC().fit(X_train,y_train)
from sklearn.metrics import precision_score,recall_score,accuracy_score

y_pred=svm.predict(X_test)

accuracyScore=accuracy_score(y_test,y_pred)
precisionScore=precision_score(y_test,y_pred)
recallScore=recall_score(y_test,y_pred)

#print(accuracyScore,precisionScore,recallScore)

from sklearn.metrics import confusion_matrix

the_svm=SVC(C=1e9,gamma=1e-07).fit(X_train,y_train)
svm_predicted=the_svm.decision_function(X_test)>-50
#print(svm_predicted)
confusion=confusion_matrix(y_test,svm_predicted)

#print(confusion)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve,roc_curve
import matplotlib.pyplot as plt 
lr=LogisticRegression().fit(X_train,y_train)
prob=lr.decision_function(X_test)
precision,recall,_=precision_recall_curve(y_test,prob)

fpr,tpr,_=roc_curve(y_test,prob)

precision_index=np.argwhere(precision==0.75)
recall_specified=recall[precision_index]
print(recall_specified)

fpr_index=np.argwhere(fpr==0.16)
tpr_specified=tpr[fpr_index]
print(tpr_specified)

plt.subplot(1,2,1)
plt.plot(precision,recall)
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.vlines(0.75,0,1)
plt.subplot(1,2,2)
plt.plot(fpr,tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.vlines(0.16,0,1)
plt.tight_layout()
#plt.show()

answers=(0.825,0.9145)

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

lrg=LogisticRegression()
grid_params={'penalty':['l1','l2'],'C':[0.01,0.1,1,10,100]}


clf=GridSearchCV(lrg,param_grid=grid_params,scoring='recall')
clf.fit(X_train,y_train)
results=clf.cv_results_

ans=np.array(results['mean_test_score'].reshape(5,2))

print(ans)

