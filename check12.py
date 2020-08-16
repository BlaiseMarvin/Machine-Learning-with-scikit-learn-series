import pandas as pd
data={'Day':[1,2,3,4,5],'People':['Matthew','Mark','Luke','John','Acts'],'Country':['Uganda','Kenya','Tanzania','Rwanda','Burundi']}
data_dataframe=pd.DataFrame(data)
coeffi=[5,-21,-18]



f=[1,2,3,4,5]


for e in sorted(list(zip(list(data_dataframe),coeffi)),key=lambda e: -abs(e[1])):
    if e[1]!=0:
        print("\t {} {}".format(e[0],e[1]))