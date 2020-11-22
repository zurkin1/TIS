from fastai import *
from fastai.vision import *
#from fastai.tabular import *
#from fastai.metrics import error_rate
from fastai.vision.models.wrn import wrn_22
#from fastai.distributed import *
from fastai.callbacks import *
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, mean_squared_error
import argparse
import os
import numpy as np
from tqdm import tqdm
from tis import auto_gpu_selection


tqdm.pandas()
base_dir = '/home/dsi/zurkin/data/data23'


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()


#python -m torch.distributed.launch --nproc_per_node=4 tis2.py
#parser = argparse.ArgumentParser()
#parser.add_argument("--local_rank", type=int)
#args = parser.parse_args()
#torch.cuda.set_device(auto_gpu_selection()) # args.local_rank) # 3
#torch.distributed.init_process_group(backend='nccl', init_method='env://')
#ssh dgx01 'gpustat'


stand = lambda x: (x-5.72)/14.1
func = lambda x: stand(float(x.parts[-1].split('_')[0])) # Standardization.
#func = lambda x: float(x.parts[-1].split('_')[0])/10 # Scale.


def tis_cdf(x):
    'CDF of the TIS data.'
    df = pd.read_csv(base_dir+'/data_v2.csv', index_col=0)
    tis = float(x.parts[-1].split('_')[0])
    return len(df.loc[df.score<=tis])/len(df)
    

def train_model():
    img_dir = Path(base_dir)
    #df = pd.read_csv(base_dir+'/data_v2.csv', index_col=0)
    #data_cl = ImageDataBunch.from_folder(img_dir, bs=8,
    #        ds_tfms=get_transforms(do_flip=True, flip_vert=True, max_rotate=180.0, max_lighting=0.4, p_affine=0.7, p_lighting=0.7),
    #        size=512).normalize(imagenet_stats)
                    #.transform(get_transforms())\ #do_flip=True, flip_vert=True, max_rotate=180.0, max_lighting=0.4, p_affine=0.7, p_lighting=0.7), size=512)\
    data_re = ImageList.from_folder(img_dir)\
                    .split_by_rand_pct()\
                    .label_from_func(func, label_cls=FloatList)\
                    .transform(get_transforms(do_flip=True, flip_vert=True, max_rotate=180.0, max_lighting=0.4, p_affine=0.7, p_lighting=0.7), size=512)\
                    .databunch(bs=8)\
                    .normalize(imagenet_stats)
    learn = cnn_learner(data_re, models.alexnet) # , metrics=[AUROC()]) #.to_distributed(args.local_rank)
    #learn.model = nn.Sequential(learn.model[0],
    #                            nn.Sequential(
    #                                          *list(children(learn.model[1])),
    #                                          nn.Sigmoid()))
    #learn.split([learn.model[0][0][5], learn.model[0][1], learn.model[1][3]])
    #learn.freeze_to(2)
    #learn.load('/home/dsi/zurkin/data20/models/model')
    #learn.loss_func = BCEWithLogitsFlat() # CrossEntropyFlat(weight=torch.FloatTensor([1.,1.6]).cuda()) #[1.,1.788]).cuda()) # 0.64/0.64, 0.64/0.36.
    #learn.data = data
    #learn.fit(0, lr=0) # Hack for init metric.
    #print(learn.validate(metrics=[AUROC()]))

    #learn.lr_find()
    #fig = learn.recorder.plot(return_fig=True, suggestion=True)
    #fig.savefig('fig')
    #fig = learn.recorder.plot_lr(return_fig=True)
    #fig.savefig('lr')
    #print(learn.recorder.min_grad_lr)
    
    learn.fit_one_cycle(cyc_len=30, max_lr=9e-4, wd=0.3, div_factor=10) # callbacks=[SaveModelCallback(learn, name=f'model', monitor='auroc')]) # 3e-3 div_factor=10, max_lr=slice(1e-7, 1e-4), pct_start=0.5
    #learn.freeze_to(1)
    #learn.fit_one_cycle(cyc_len=20, max_lr=9e-4, wd=0.3, div_factor=10) # callbacks=[SavedModelCallback(learn, name=f'model', monitor='auroc')]) # 3e-3 div_factor=10, max_lr=slice(1e-7, 1e-4), pct_start=0.5i
    #learn.load(base_dir+f'/models/model')
    return learn


def test_model(learn):
    #data = ImageDataBunch.from_folder(Path(base_dir +'_test/'),
            #bs=2,
            #valid_pct=0.99,
    #        size=700).normalize(imagenet_stats)
    data = ImageList.from_folder(Path(base_dir+'_test'))\
                    .split_by_rand_pct(valid_pct=0.99)\
                    .label_from_func(tis_cdf, label_cls=FloatList)\
                    .databunch(no_check=True)\
                    .normalize(imagenet_stats)
    learn.data = data
    print(learn.validate())


def test_model2(learn=None):
    df = pd.read_csv(base_dir+'/data_v2.csv', index_col=0)
    #test = pd.read_csv(base_dir+'_test/temp1.csv', index_col=0)[['image_name', 'group']]
    #df = pd.merge(df, test, on = 'image_name')
    #df['image_name'] = df.name.apply(lambda x: x[5:12])
    df['pred'] = -1
    #df['score'] = (df.score-5.7)/1.4
    len_df = len(df)
    #def cdf_score(x):
    #    return len(df.loc[df.score<=x])/len_df
    #df['cdf_score'] = df.score.apply(cdf_score)
    df['zscore'] = df.score.apply(stand)
    avg_score = np.mean(df.zscore)
    df['bin_score'] = df.zscore.apply(lambda x: 0 if x <= avg_score else 1) # 0.57238 0.56178
    
    preds = []
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    #for g in ['low', 'high']:
    for f in Path(base_dir+'_test/').iterdir():
        img = open_image(f)
        for i in range(3):
            img.data[i,:,:] -= mean[i]
            img.data[i,:,:] /= std[i]
        img.resize(torch.Size([img.shape[0],512,512]))
        x1,x2,x3 = learn.predict(img)
        #preds.append(int(x))
        start = str(f).find('TCGA')
        df.loc[df.image == str(f)[start:], 'pred'] = x1.data[0] # float(x3[0]) #1-float(x2.data) #x3[0])
        #df.loc[df.image == str(f)[start:], 'bin_score'] = 0 if g == 'low' else 1
    df = df.loc[df.pred > -1]
    #df['bin_pred'] = df.pred.apply(lambda x: 0 if x < 0.6 else 1)

    #df2 = df.groupby('image_name').agg({'pred':np.median, 'bin_score':np.max})
    print(roc_auc_score(df.bin_score, df.pred), mean_squared_error(df.zscore, df.pred)) # , roc_auc_score(df2.bin_score, df2.pred))
    df.to_csv(base_dir+'/temp.csv')
    df['pred'] = avg_score # 0.56178 #Baseline model.
    print(mean_squared_error(df.zscore, df.pred))
    #df2.to_csv(base_dir+'data_v3d.csv')


def use_model(learn):
    base_dir = '/home/dsi/rimonshy/data30/part_b'
    def process_image(x):
        data = ImageDataBunch.from_df(Path(base_dir), x, label_col='bin_score', valid_pct=0, bs=20, size=750).normalize()
        data.valid_dl = data.train_dl
        learn.data = data
        x1, x2 = learn.get_preds()
        diff = len(x) - len(x1) #Padding.
        r = np.concatenate((x1[:,0], [x1[:,0].mean()]*diff))
        if data.classes[0] == 'low':
            r = 1 - r
        x['pred'] = r
        return x


    df = pd.read_csv(base_dir+'.csv')
    df['image_name'] = df.name.apply(lambda x: x[5:12])
    df['bin_score'] = df.score.apply(lambda x: 0 if x < 5.7 else 1)
    test = pd.read_csv('/home/dsi/zurkin/data22/temp1.csv', index_col=0)[['image_name', 'group']]
    df = pd.merge(df, test, on = 'image_name')
    df2 = df.groupby('image_name').progress_apply(process_image)
    df2.to_csv(f'part_b_out.csv')
 

def train_histograms():
    df = pd.read_csv('histdata.csv')
    df = df.groupby('image_name').agg({'score':np.max, 'pred':np.mean}) #histogram})
    #df['pred'] = df.pred.apply(lambda x: x[0])
    #groups = pd.read_csv(base_dir+'/data_v2.csv', index_col=0)[['image_name', 'group']]
    #df = pd.merge(df, groups, on = 'image_name')
    #data = pd.DataFrame(df.pred.tolist())
    df['label'] = df.score.apply(lambda x: 0 if x < 5.7 else 1)
    #train_idx = df.loc[df.group == 'train'].index
    #val_idx = df.loc[df.group == 'validate'].index
    #test_idx = df.loc[df.group == 'test'].index
    #train = TabularDataBunch.from_df(Path('.'), data.iloc[train_idx|val_idx], dep_var='label', valid_idx=val_idx, cont_names=[0,1,2,3,4,5,6,7,8,9], bs=8)
    #learn = tabular_learner(train, layers=[100,50,20], metrics=[AUROC()], use_bn=True, ps=[0.3,0.2,0.1], y_range=[0,1])
    #learn.lr_find()
    #fig = learn.recorder.plot(return_fig=True, suggestion=True)
    #fig.savefig('fig')
    #fig = learn.recorder.plot_lr(return_fig=True)
    #fig.savefig('lr')
    #print(learn.recorder.min_grad_lr)
    #learn.fit_one_cycle(30, 1e-3)

    #data = (data - data.min())/(data.max() - data.min())
    #model = LogisticRegression(random_state=0).fit(data.iloc[train_idx|val_idx], df.iloc[train_idx|val_idx].label)
    #pred = model.predict_proba(data.iloc[test_idx])

    #pred = df.iloc[test_idx].pred #idxmax(axis=1)
    print(roc_auc_score(df.label, df.pred))


if __name__ == '__main__':
    learn = train_model()
    test_model2(learn)
    #test_model(learn)
    #use_model(learn)
    #train_histograms()
