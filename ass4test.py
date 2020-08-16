import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score,auc
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve
import numpy as np

df=pd.read_csv('train.csv',encoding="ISO-8859-1",dtype={'zip_code': str, 'non_us_str_code': str, 'grafitti_status': str, 'violator_name':str, 'mailing_address_str_number': str})

df.compliance=df.compliance.fillna(value=-1)
df=df[df['compliance']!=-1]

X=df.loc[:,'ticket_id':'judgment_amount']
X['grafitti_status']=df['grafitti_status']

y=df.iloc[:,-1]

X=X.fillna('')
X_1 = X.select_dtypes(include=[object])
le=LabelEncoder()

X_2 = X_1.apply(le.fit_transform)

float_columns = ['ticket_id', 'violation_street_number', 'fine_amount', 'admin_fee', 'state_fee', 'late_fee','discount_amount', 'clean_up_cost', 'judgment_amount']

X_2[float_columns]=X[float_columns]

X_2_train, X_2_test, y_train, y_test = train_test_split(X_2, y, random_state=0)
grd = GradientBoostingClassifier()
grd.fit(X_2_train, y_train)

y_pred=grd.predict_proba(X_2_test)[:,1]

df

