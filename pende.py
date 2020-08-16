import pandas as pd 

xyz_web={"Day":[1,2,3,4,5,6],"Visitors":[1000,700,6000,1000,400,350],"Bounce_rate":[20,20,23,15,10,34]}

df= pd.DataFrame(xyz_web)

df1=pd.DataFrame({"HPI":[80,90,70,60],"Int_rate":[2,1,2,3],"IND_GDP":[50,45,45,67]},
                  index=[2001,2002,2003,2004]  )

df2=pd.DataFrame({"HPI":[80,90,70,60],"Int_rate":[2,1,2,3],"IND_GDP":[50,45,45,67]},
                   index=[2005,2006,2007,2008] )



merge=pd.merge(df1,df2,on="Int_rate")


df3=pd.DataFrame({"Int_rate":[2,1,2,3],"IND_GDP":[50,45,45,67]},index=[2001,2002,2003,2004])
df4=pd.DataFrame({"Low_Tier_HPI":[50,45,67,34],"Unemployment":[1,3,5,6]},index=[2001,2003,2004,2005])

joined=df3.join(df4)
print(joined)


