import pandas as pd 

data=pd.read_table('fruit_data_with_colors.txt')
data.to_html('edu.html')