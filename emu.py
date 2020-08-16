from sklearn.datasets import load_breast_cancer
import pandas as pd
cancer=load_breast_cancer()
def answer_one():
   
   
    cancer_dataframe=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
    Target=cancer['target']
    cancer_dataframe['target']=Target
    
    return cancer_dataframe
cancerdf = answer_one()
X=cancerdf[cancerdf.columns[:-1]]
y=cancerdf.target
print(X.shape)
print(y.shape)



