import os
from shutil import copyfile
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import shutil




def filter_data():
    dir_in = '/home/dsi/zurkin/data/data31/'
    dir_out = '/home/dsi/zurkin/data/data31a_test/'

    df = pd.read_csv('/home/dsi/zurkin/data/data31a/test.csv',index_col=['image'])
    lst = list(df.index.values)
    print(lst)
    # print(df.loc['TCGA-AC-A3YJ-01Z-00-DX1.8E665F69-FD8C-419A-871F-3AEE2E5A3A60.251-7.jpg']['class'])
    for filename in os.listdir(dir_in):
        #start_name = filename.split(".")[0]
        #print(filename)
        if filename in lst:
            #print(df.loc[filename]['class'])
            copyfile(dir_in+filename, dir_out +df.loc[filename]['class']+"/"+filename)
            


def split_data():
    df = pd.read_csv('/home/dsi/zurkin/data27/1045.csv')[['image', 'score']]

    df['class']= ['low' if x<5 else 'high' for x in df['score']]

    #train, validate, test = np.split(df.sample(frac=1), [int(.80*len(df)), int(.9*len(df))])  #split to: train 80%, test 10%, validation 10%
    df['image_name'] = df.image.apply(lambda x: x[5:12])
    image_num = df.image_name.unique()
    #print(image_num)
    train, test, validate = np.split(image_num, [int(.70*len(image_num)), int(.85*len(image_num))])
    #train, test, validate = np.split(image_num, [int(.80*len(image_num)), int(len(image_num))])

    train = df[df.image_name.isin(train)][['image','score', 'class']].sample(frac=1)
    test = df[df.image_name.isin(test)][['image', 'score','class']].sample(frac=1)
    validate = df[df.image_name.isin(validate)][['image', 'score','class']].sample(frac=1)

    # print(train.head())
    # print(f'Train: {len(train)}, Test: {len(test)}')
    return train, test, validate

def analyse():
    df = pd.read_csv('/home/dsi/zurkin/data20/data_v3e.csv', index_col=0)
    #df['image_name'] = df.image.apply(lambda x: x[5:12])
    #df['bin_pred'] = df.pred.apply(lambda x: 0 if x < 0.6 else 1)
    #df2 = df.groupby('image_name').agg({'pred':np.median, 'bin_score':np.max})
    #print(df2)
    #df2 = df.groupby(['image_name']).mean()
    #df2['bin_pred'] = df2.pred.apply(lambda x: 0 if x < 0.5 else 1)
    # AUC_patch = roc_auc_score(df['bin_score'], df['pred'])
    df2 = df.groupby('image_name').agg({'pred':np.median, 'bin_score':np.max})
    print(roc_auc_score(df2.bin_score, df2.pred))

    #print(AUC_image)


def prepare_data(csv_file):
    df = pd.read_csv(csv_file)
    #df['int_score'] = df.score.round()
    #df_5 = df.loc[df.int_score == 5].sample(4300).copy()
    #df = df.loc[(df.int_score == 6) & (df.group == 'train')].copy()
    #df = pd.concat([df_5, df_not_5])
    #df = pd.read_csv('/gdrive/My Drive/TIS/public/set_1_2_3_1000.csv')[['image', 'score']]
    #(train, test) = train_test_split(df, test_size=0.25, random_state=42)
    #df['bin_score'] = df.score.apply(lambda x: to_categorical(0 if x < 7 else 1, num_classes=2))
    df['bin_score'] = df.score.apply(lambda x: 'low' if x < 5.6 else 'high')
    df['size'] = 0

 #   df['group'] = 'nogroup'
    
    df['image_name'] = df.image.apply(lambda x: x[5:12])
    '''
    df = df.sample(frac=1)
    image_num = df.image_name.unique()
    train, validate, test = np.split(image_num, [int(.70*len(image_num)), int(.85*len(image_num))])
    df.loc[df.image_name.isin(train), 'group'] = 'train' #[['image', 'bin_score']].sample(frac=1)
    df.loc[df.image_name.isin(test), 'group'] = 'test' #[['image', 'bin_score']].sample(frac=1)
    df.loc[df.image_name.isin(validate), 'group'] = 'validate' #[['image', 'bin_score']].sample(frac=1)

    dest_dir = '/home/dsi/zurkin/data27/all_train_test'
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
        for d in ['/train/', '/validate/', '/test/']:
            os.mkdir(dest_dir+d)
            for l in ['low', 'high']:
                os.mkdir(dest_dir+d+l)
    
    #df2 = pd.read_csv('data3/data.csv')
    #df2['image_name'] = df2.image.apply(lambda x: x[5:12])
    #df2 = set(df2.image_name)
    #df2 = df.groupby(['image_name'], as_index=False).apply(lambda x: x if len(x)< i+1 else x.iloc[[i]]).reset_index(level=0, drop=True)
    df['size'] = df.image.apply(lambda x: os.path.getsize(root_path+'all/'+x))
    df2 = df.copy().sort_values('size', ascending=False).drop_duplicates('image_name')
    #df.to_csv(root_path+'data45a.csv')

    for ind, row in df2.iterrows():
        if os.path.isfile(root_path+'large/'+row['image_name']+'.jpg'):
        #shutil.copy(root_path+'all/'+row[0], f'{dest_dir}/{row[4]}/{row[2]}/{row[0]}') # {row[5]}/{row[3]}/{row[0]}')
            shutil.copy(root_path+'large/'+row['image_name']+'.jpg', f'{dest_dir}/{row[4]}/{row[2]}/{row[0]}') # {row[5]}/{row[3]}/{row[0]}')

    # for ind, row in df.iterrows():
    #    if row[4] == 'test':
    #        shutil.copy(root_path+row[0], f'{dest_dir}_test_all/{row[2]}/{row[0]}')

    #df.to_csv(dest_dir+'/data_v1.csv')
    dest_dir = '/home/dsi/zurkin/data21_all/'

    df.to_csv(dest_dir+'/data_v2.csv')
'''

# train, test, validate = split_data()
# train.to_csv("/home/dsi/zurkin/data27/train.csv", index=False)
# test.to_csv("/home/dsi/zurkin/data27/test.csv", index=False)
# validate.to_csv("/home/dsi/zurkin/data27/validate.csv", index=False)
#prepare_data(root_path='/home/dsi/zurkin/data27/',csv_file='/home/dsi/zurkin/data27/1045.csv')

#analyse()
#prepare_data('/home/dsi/zurkin/data21_all/1045.csv')
'''
df = pd.read_csv('/home/dsi/zurkin/data/data31.csv')[['image', 'score']]
df = df.sample(frac=1) #shaffle
df['class']= ['low' if x<6 else 'high' for x in df['score']]

    #train, validate, test = np.split(df.sample(frac=1), [int(.80*len(df)), int(.9*len(df))])  #split to: train 80%, test 10%, validation 10%
df['image_name'] = df.image.apply(lambda x: x[5:12])
image_num = df.image_name.unique()
    #print(image_num)
train, test, validate = np.split(image_num, [int(.70*len(image_num)), int(.85*len(image_num))])
    #train, test, validate = np.split(image_num, [int(.80*len(image_num)), int(len(image_num))])

train = df[df.image_name.isin(train)][['image','score', 'class']].sample(frac=1)
test = df[df.image_name.isin(test)][['image', 'score','class']].sample(frac=1)
validate = df[df.image_name.isin(validate)][['image', 'score','class']].sample(frac=1)

train.to_csv("/home/dsi/zurkin/data/data31a/train.csv", index=False)
test.to_csv("/home/dsi/zurkin/data/data31a/test.csv", index=False)
validate.to_csv("/home/dsi/zurkin/data/data31a/validate.csv", index=False)
'''
filter_data()