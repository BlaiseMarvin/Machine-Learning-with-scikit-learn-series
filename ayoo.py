import pandas as pd 
nomres={'yi':[1,2,3,4,5],'ya':[6,7,8,9,10]}
df=pd.DataFrame(nomres)
print(df)
print(df.mean())
print(df.mean().values.reshape(1,-1))