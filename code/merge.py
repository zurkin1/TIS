from os import listdir
from os.path import isfile, join
import pandas as pd


mypath = './kallisto/'
onlyfiles = [f for f in listdir(mypath) if not isfile(join(mypath, f))]
df = pd.DataFrame()
for dir in onlyfiles:
    ab = pd.read_csv(mypath+dir+'/abundance.tsv', sep='\t', index_col='target_id')
    dir = dir.split('_')[0]
    ab.columns = [dir+'_length', dir+'_eff_length', dir+'_est_counts', dir+'_tpm']
    df = pd.concat([df, ab], axis=1)

df.to_csv('merged.csv')