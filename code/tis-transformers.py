from fastai.vision import *
#from fastai.metrics import error_rate
from fastai.vision.models.wrn import wrn_22
#from fastai.distributed import *
from fastai.callbacks import *
from pathlib import Path
from sklearn.metrics import roc_auc_score
import argparse
import os

#python -m torch.distributed.launch --nproc_per_node=4 tis2.py
#parser = argparse.ArgumentParser()
#parser.add_argument("--local_rank", type=int)
#args = parser.parse_args()
#torch.cuda.set_device(5) #args.local_rank) # 3
#torch.distributed.init_process_group(backend='nccl', init_method='env://')

base_dir = '/home/dsi/rimon/data15/'
img_dir = Path(base_dir)
#csv_file = '/home/dsi/zurkin/set_1-7_only1.csv'
# data = ImageDataBunch.from_csv(p_imgs, csv_labels=csv_file,
#                                fn_col="image", label_col="binaryclass", bs=12, 
#                                ds_tfms=get_transforms(do_flip=True, flip_vert=True,
#                                                       max_rotate=180.0, max_lighting=0.2,
#                                                       p_affine=0.75, p_lighting=0.75)).normalize()

test = ImageList.from_folder('/home/dsi/zurkin/data21/')

data = ImageDataBunch.from_folder(img_dir, #bs=16,
        train=Path(base_dir+'train'),
        valid=Path(base_dir+'valid'),
        ds_tfms=get_transforms(do_flip=True, flip_vert=True, max_rotate=180.0, max_lighting=0.2, p_affine=0.75, p_lighting=0.75),
        size=500).normalize().add_test(test) #size=500

for cyc in range(1):
    print(f'Cycle {cyc}.')
    learn = cnn_learner(data, models.alexnet, metrics=[AUROC()]) #.to_distributed(args.local_rank)
#learn = load_learner('/home/dsi/zurkin/data2', 'tis.pkl')
    if os.path.isfile(base_dir+'models/bestmodel.pth'):
        learn.load(base_dir+'models/bestmodel')
#learn.loss_func = CrossEntropyFlat(weight=torch.FloatTensor([1.,1.7]).cuda()) # 0.64/0.64, 0.64/0.36.
    learn.loss_func = CrossEntropyFlat(weight=torch.FloatTensor([2.4,1.]).cuda()) #[1.,1.788]).cuda()) # 0.64/0.64, 0.64/0.36.

#learn.lr_find()
#fig = learn.recorder.plot(return_fig=True, suggestion=True)
#fig.savefig('fig')
#print(learn.recorder.min_grad_lr)

#    learn.fit_one_cycle(2, max_lr=1e-4, wd=0.7, pct_start=0.5)
    learn.unfreeze()
    learn.fit_one_cycle(cyc_len=1, max_lr=1e-4, wd=0.7, callbacks=[SaveModelCallback(learn, monitor='auroc')]) # 3e-3 div_factor=10, max_lr=slice(1e-6, 1e-4)    
#learn.load(base_dir+'models/bestmodel')
#learn.save('tis')
#learn.export('tis.pkl')
#fig = learn.recorder.plot_lr(return_fig=True)
#fig.savefig('lr')


'''
base_dir='/home/dsi/zurkin/data15_test/'
df = pd.read_csv(base_dir+'test.csv', index_col=0)
predictions, *_ = learn.get_preds(DatasetType.Test)
labels = np.argmax(predictions, 1)
df['category'] = labels
df.to_csv(base_dir+'temp2.csv')
'''
'''
#df['image_name'] = df.image.apply(lambda x: x[5:12])
df['bin_score'] = -1 # df.score.apply(lambda x: 0 if x < 6. else 1)
df['pred'] = -1
preds = []
score = 0
for g in ['low', 'high']:
    for f in Path(base_dir+g).iterdir():
        x1,x2,x3 = learn.predict(open_image(f))
        #preds.append(int(x))
        start = str(f).find('TCGA')
        df.loc[df.image == str(f)[start:], 'pred'] = float(x3[0]) #1-float(x2.data) #x3[0])
        df.loc[df.image == str(f)[start:], 'bin_score'] = score
    score += 1
df = df.loc[df.pred > -1]
#df['bin_pred'] = df.pred.apply(lambda x: 0 if x < 0.6 else 1)
#df2 = df.groupby('image_name').agg({'pred':np.mean, 'bin_score':np.max})
print(roc_auc_score(df.bin_score, df.pred)) #, roc_auc_score(df2.bin_score, df2.pred))
df.to_csv(base_dir+'temp1.csv')
#df2.to_csv(base_dir+'temp2.csv')"
'''