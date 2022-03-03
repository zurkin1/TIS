import pandas as pd
import os

df = pd.read_csv('kaggle_2021.csv')
files = [x.split('.')[0] for x in os.listdir('publichpa_p/') if '.jpg' in x]
df = df.loc[df.ID.isin(files)]
df.to_csv('kaggle.csv', index=False)
