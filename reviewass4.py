import pandas as pd 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

df=pd.read_csv('train.csv',encoding='ISO-8859-1',low_memory=False)

df.index=df['ticket_id']

features_names=['fine_amount','admin_fee','state_fee','late_fee','discount_amount','clean_up_cost','judgment_amount']

X= df[features_names]

df.compliance=df.compliance.fillna(value=-1)

df=df[df['compliance']!=-1]

y=df['compliance']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

print(X_train)



