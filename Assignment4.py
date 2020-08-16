import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score,auc
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve
from sklearn.neural_network import MLPClassifier
df=pd.read_csv('train.csv',encoding="ISO-8859-1",low_memory=False)
df.index=df['ticket_id']
df.compliance=df.compliance.fillna(value=-1)
df=df[df.compliance!=-1]



features_name=['fine_amount','violation_street_number','admin_fee','state_fee','late_fee','judgment_amount','discount_amount','clean_up_cost']

X= df[features_name]





y=df.compliance

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)



clf=GradientBoostingClassifier(n_estimators=100,max_depth=3,learning_rate=0.1).fit(X_train,y_train)

y_score=clf.predict_proba(X_test)[:,1]

df_test=pd.read_csv('test.csv',encoding="ISO-8859-1")
df_test.index=df_test['ticket_id']

X_predict=clf.predict_proba(df_test[features_name])

ans=pd.Series(data=X_predict[:,1],index=df_test['ticket_id'],dtype='float32')

print(ans)
















