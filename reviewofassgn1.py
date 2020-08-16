import numpy as np 
import pandas as pd 
from sklearn.datasets import load_breast_cancer

cancer=load_breast_cancer()

#qn one: how many features does the breast cancer dataset have

def question_zero():
    return len(cancer['feature_names'])

def answer_one():
    cancer_dataframe=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
    Target=cancer['target']
    cancer_dataframe['target']=Target

    return cancer_dataframe


""""""
cancerdf=answer_one()
instances=cancerdf.target.value_counts(ascending=True)
instances.index="malignant benign".split()
""""""

#now split the dataframe into data:X and labels: y
cancerdf=answer_one()
X=cancerdf[cancerdf.columns[:-1]]
y=cancerdf['target']


