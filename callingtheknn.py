from sklearn.model_selection import train_test_split
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier

fruits=pd.read_table('fruit_data_with_colors.txt')

#create a dictionary that has labels and fruit names

look_up_fruit_name=dict(zip(fruits.fruit_label.unique(),fruits.fruit_name.unique()))

X=fruits[['mass','width','height']]
y=fruits['fruit_label']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)

print(knn.score(X_test,y_test))

fruit_prediction=knn.predict([[20,4.3,5.5]])
print(look_up_fruit_name[fruit_prediction[0]])




