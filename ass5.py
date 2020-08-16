import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


mush_df = pd.read_csv('mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

# use the variables X_train2, y_train2 for Question 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
X_subset = X_test2
y_subset = y_test2


from sklearn.svm import SVC
from sklearn.model_selection import validation_curve

gamma=np.logspace(-4,1,6)
training_scores,testing_scores=validation_curve(SVC(),X_subset,y_subset,param_name='gamma',param_range=gamma,scoring='accuracy')
the_scores=(training_scores.mean(axis=1),testing_scores.mean(axis=1))

 
    
df=pd.DataFrame({'Training_Scores':the_scores[0],'Test_Scores':the_scores[1]})
df['Gamma']=np.logspace(-4,1,6)
df['Difference']=df['Training_Scores']-df['Test_Scores']
df.set_index('Gamma',inplace=True)


df=df.sort_values(['Difference'])

print(df)

Good_Generalization=df.index[0]

df=df.sort_values(['Difference'],ascending=False)

print(df)

    
Overfitting=df.index[0]
df=df.sort_values(['Training_Scores'])

print(df)
    
Underfitting=df.index[0]

print((Underfitting,Overfitting,Good_Generalization))


    
   

