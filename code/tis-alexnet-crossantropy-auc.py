from fastai import *
from fastai.vision import *
#from fastai.metrics import error_rate
from fastai.vision.models.wrn import wrn_22
#from fastai.distributed import *
from fastai.callbacks import *
from pathlib import Path
from sklearn.metrics import roc_auc_score
import argparse
import os
import pandas as pd
import numpy as np

path = '/home/dsi/rimon/data17/all'
path = Path(path)
df = pd.read_csv('/home/dsi/zurkin/data17/1027.csv')[['image', 'score']]

df['class']= ['low' if x<6 else 'high' for x in df['score']]

#train, validate, test = np.split(df.sample(frac=1), [int(.80*len(df)), int(.9*len(df))])  #split to: train 80%, test 10%, validation 10%
df['image_name'] = df.image.apply(lambda x: x[5:12])
image_num = df.image_name.unique()
#print(image_num)
train, test, validate = np.split(image_num, [int(.70*len(image_num)), int(.85*len(image_num))])
#train, test, validate = np.split(image_num, [int(.80*len(image_num)), int(len(image_num))])

df_train = df[df.image_name.isin(train)][['image','score', 'class']].sample(frac=1)
df_test = df[df.image_name.isin(test)][['image', 'score','class']].sample(frac=1)
df_validate = df[df.image_name.isin(validate)][['image', 'score','class']].sample(frac=1)

train = ImageList.from_df(df_train, path)
test = ImageList.from_df(df_test, path)

data = ImageDataBunch.from_df(path, df_train, ds_tfms=get_transforms(), size=500 ).normalize()
data.add_test(test)

learn = cnn_learner(data, models.alexnet, metrics=[AUROC()])
if os.path.isfile(path+'models/bestmodel.pth'):
        learn.load(path+'models/bestmodel')
learn.loss_func = CrossEntropyFlat(weight=torch.FloatTensor([2.4,1.]).cuda())

learn.fit_one_cycle(2)
learn.unfreeze()
learn.fit_one_cycle(cyc_len=2, max_lr=1e-4, wd=0.7, callbacks=[SaveModelCallback(learn, monitor='auroc')])

predictions, *_ = learn.get_preds(DatasetType.Test)
labels = np.argmax(predictions, 1)
df_test['category'] = labels

df_test.to_csv('/home/dsi/zurkin/data17/temp.csv')
