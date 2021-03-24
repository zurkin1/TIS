import numpy as np
import pandas as pd
import os
import shutil


df = pd.read_csv('tis_skcm.csv')
#df['int_score'] = df.score.round()
#df_5 = df.loc[df.int_score == 5].sample(4300).copy()
#df = df.loc[(df.int_score == 6) & (df.group == 'train')].copy()
#df = pd.concat([df_5, df_not_5])
#df = pd.read_csv('/gdrive/My Drive/TIS/public/set_1_2_3_1000.csv')[['image', 'score']]
#(train, test) = train_test_split(df, test_size=0.25, random_state=42)
#df['bin_score'] = df.score.apply(lambda x: to_categorical(0 if x < 7 else 1, num_classes=2))
df['bin_score'] = df.score.apply(lambda x: 'low' if x < 5.6 else 'high')
df['size'] = 0
df['group'] = 'nogroup'
df['image'] = df.image.apply(lambda x: str(x).replace('.','-'))
df['image_name'] = df.image.apply(lambda x: x[5:12])
df = df.sample(frac=1)
image_num = df.image_name.unique()
#train, validate, test = np.split(image_num, [int(.70*len(image_num)), int(.85*len(image_num))])
train, test = np.split(image_num, [int(.85*len(image_num))])
df.loc[df.image_name.isin(train), 'group'] = 'train' #[['image', 'bin_score']].sample(frac=1)
df.loc[df.image_name.isin(test), 'group'] = 'test' #[['image', 'bin_score']].sample(frac=1)


root1 = './skcm_prob/'

for sub1, dirs1, files1 in os.walk(root1):
    for root2 in dirs1:
        max_size = 0
        for sub2, dirs2, files2 in os.walk(root1+root2):
            for fil in files2:
              size = os.stat(os.path.join(root1+root2,fil)).st_size
              if size>max_size:
                max_size = size
                max_file = fil
            #print(max_file)
            if root2[5:12] in df.image_name.values:
                dest = df.loc[df.image_name == root2[5:12]].group.values[0]
                shutil.copy(root1+root2+'/'+fil, './data24/'+dest+'/'+root2[5:12])
        #print(os.path.join(subdir, file))
        #print(sub1, root2)

"""
dest_dir = './data24'
if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)
    for d in ['/train/', '/validate/', '/test/']:
        os.mkdir(dest_dir+d)
        #for l in ['low', 'high']:
        #    os.mkdir(dest_dir+d+l)
    
#df2 = pd.read_csv('data3/data.csv')
#df2['image_name'] = df2.image.apply(lambda x: x[5:12])
#df2 = set(df2.image_name)
#df2 = df.groupby(['image_name'], as_index=False).apply(lambda x: x if len(x)< i+1 else x.iloc[[i]]).reset_index(level=0, drop=True)
df['size'] = df.image.apply(lambda x: os.path.getsize(root_path+x))
df2 = df.copy().sort_values('size', ascending=False).drop_duplicates('image_name')
#df.to_csv(root_path+'data45a.csv')

for ind, row in df2.iterrows():
    if os.path.isfile(root_path+row['image']):
        shutil.copy(root_path+row['image'], f'{dest_dir}/{row.group}/{row.bin_score}/{str(round(row.score,2))}_{row.image}') # {row[5]}/{row[3]}/{row[0]}')

#for ind, row in df.iterrows():
#    if row[4] == 'test':
#        shutil.copy(root_path+row[0], f'{dest_dir}_test_all/{row[2]}/{row[0]}')

#df.to_csv(dest_dir+'/data_v1.csv')
df2.to_csv(dest_dir+'/data_v2.csv')
"""
