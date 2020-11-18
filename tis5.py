from fastai.vision.all import *
from fastai.metrics import error_rate
from fastai.distributed import *
from pathlib import Path
from sklearn.metrics import roc_auc_score
import argparse


torch.cuda.set_device(0) #args.local_rank) # 3
base_dir = '/home/dsi/zurkin/data20/'
img_dir = Path(base_dir)
data = ImageDataLoaders.from_folder(img_dir, train='train', valid='validate', bs=16,
       item_tfms=Resize(512), batch_tfms=aug_transforms(size=512)) #size=500

learn = cnn_learner(data, resnet34, metrics=[RocAucBinary()]) #.to_distributed(args.local_rank)
#learn.load(base_dir+'models/model')
#learn.to_fp16()
learn.fit_one_cycle(n_epoch=30, max_lr=5e-3, wd=0.4, div_factor=10, cbs=[SaveModelCallback(monitor='roc_auc_score')]) # 3e-3 

"""
df = pd.read_csv('/home/dsi/zurkin/data7/1000/data_v2.csv', index_col=0)
base_dir='/home/dsi/zurkin/data12/'
df['bin_score'] = df.score.apply(lambda x: 0 if x < 6. else 1)
df['pred'] = -1
preds = []
for g in ['low', 'high']:
    for f in Path(base_dir+'/test/'+g).iterdir():
        x1,x2,x3 = learn.predict(f)
        print('.', end="")
        #preds.append(int(x))
"""
#        start = str(f).find('TCGA')
#        df.loc[df.image == str(f)[start:], 'pred'] = 1-float(x2.data) #x3[0])
#data = ImageDataLoaders.from_folder(img_dir, train='train', valid='test')
#x = learn.validate()
#print(x)
#df = df.loc[df.pred > -1]
#df['bin_pred'] = df.pred.apply(lambda x: 0 if x < 0.6 else 1)
#df2 = df.groupby('image_name').agg({'pred':np.mean, 'bin_score':np.max})
#print(roc_auc_score(df.bin_score, df.pred), roc_auc_score(df2.bin_score, df2.pred))
#df.to_csv(base_dir+'temp1.csv')
#df2.to_csv(base_dir+'temp2.csv')
