from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
dataset=load_digits()
X,y=dataset['data'],dataset['target']
X_trainmc,X_testmc,y_trainmc,y_testmc=train_test_split(X,y,random_state=0)
svm=SVC(kernel='linear').fit(X_trainmc,y_trainmc)
svm_predicted_mc=svm.predict(X_testmc)
confusion_mc= confusion_matrix(y_testmc,svm_predicted_mc)
df_mc = pd.DataFrame(confusion_mc,index=[i for i in range(0,10)],columns=[i for i in range(0,10)])

plt.figure(figsize=(5.5,4))
sns.heatmap(df_mc,annot=True)
plt.title("SVM linear kernel\n Accuracy: {0:.3f} ".format(accuracy_score(y_testmc,svm_predicted_mc)))
plt.ylabel("True Label")
plt.xlabel("Predicted Label")

#plt.show()

print("Micro-averaged precision = {:.2f} (treat instances equally) ".format(precision_score(y_testmc,svm_predicted_mc,average='micro')))
print("Macro-averaged precision= {:.2f} (treat classes equally) ".format(precision_score(y_testmc,svm_predicted_mc,average='macro')))

print("Micro-averaged f1 score ={:.2f} (treat instances equally) ".format(f1_score(y_testmc,svm_predicted_mc,average='micro')))
print("Macro-averaged f1 score ={:.2f} (treat classes equally) ".format(f1_score(y_testmc,svm_predicted_mc,average='macro')))