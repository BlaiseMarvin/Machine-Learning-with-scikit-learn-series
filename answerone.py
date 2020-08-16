def answer_two():
    
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    cancer=load_breast_cancer()
    targeti={'malignant':0,'benign':0}
    for z in cancer['target']:
        if z==0:
            targeti['malignant']+=1
        elif z==1:
            targeti['benign']+=1
    target=pd.Series([targeti['malignant'],targeti['benign']])
    index=['malignant','benign']
    
    
    return target



print(answer_two())