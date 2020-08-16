import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import style

style.use("fivethirtyeight")
df=pd.DataFrame({"Day":[1,2,3,4],"Visitors":[200,100,230,300],"Bounce_rate":[20,45,60,10]})

df.set_index("Day",inplace=True)

df.plot()

df.rename(columns={"Visitors":"Users"})
print(df)