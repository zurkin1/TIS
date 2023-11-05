# from fastai.vision import *
# #from fastai.metrics import error_rate
# from fastai.vision.models.wrn import wrn_22
# from fastai.distributed import *
# from fastai.callbacks import *
# from pathlib import Path
# from sklearn.metrics import roc_auc_score
# import argparse
# import os
# import numpy as np
# from tqdm import tqdm
# from sklearn.metrics import r2_score, mean_squared_error

# from fastai import *
# from fastai.vision import *
# #from fastai.tabular import *
# #from fastai.metrics import error_rate
# from fastai.vision.models.wrn import wrn_22
# #from fastai.distributed import *
# from fastai.callbacks import *
# from pathlib import Path
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_auc_score, mean_squared_error
# import argparse
# import os
# import numpy as np
# from tqdm import tqdm
# from tis import auto_gpu_selection


from fastai.vision.all import *
from fastai.metrics import error_rate
from fastai.distributed import *
from pathlib import Path
from sklearn.metrics import roc_auc_score, mean_squared_error
import argparse
from PIL import Image
from torchvision import transforms
import os



#python -m torch.distributed.launch --nproc_per_node=4 tis2-rimon.py
'''
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank) # 3
torch.distributed.init_process_group(backend='nccl', init_method='env://')
torch.cuda.set_device(1)
'''


#tqdm.pandas()
#base_dir = "/home/dsi/rimonshy/MS_rimon/data/BRCA/data31"

base_dir = "/home/dsi/rimon/data/data21"


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()

func = lambda x: float(x.parts[-1].split('_')[0])/10

#learn.load('/home/dsi/zurkin/data20/models/model')
#learn.loss_func = CrossEntropyFlat(weight=torch.FloatTensor([1.,1.6]).cuda()) #[1.,1.788]).cuda()) # 0.64/0.64, 0.64/0.36.
#learn.data = data
#learn.fit(0, lr=0) # Hack for init metric.
#print(learn.validate(metrics=[AUROC()]))

#learn.lr_find()
#fig = learn.recorder.plot(return_fig=True, suggestion=True)
#fig.savefig('fig')
#print(learn.recorder.min_grad_lr)



'''
df = pd.read_csv(base_dir+'/data_v2.csv', index_col=0)
base_dir='/home/dsi/zurkin/data21_test_all/'
#df['image_name'] = df.image.apply(lambda x: x[5:12])
df['bin_score'] = -1 # df.score.apply(lambda x: 0 if x < 6. else 1)
df['pred'] = -1
preds = []
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
for g in ['low', 'high']:
    for f in Path(base_dir+g+'/').iterdir():
        img = open_image(f)
        # for i in range(3):
        #     img.data[i,:,:] -= mean[i]
        #     img.data[i,:,:] /= std[i]
        # img.resize(torch.Size([img.shape[0],700,700]))
        x1,x2,x3 = learn.predict(img)
        #preds.append(int(x))
        start = str(f).find('TCGA')
        df.loc[df.image == str(f)[start:], 'pred'] = float(x3[0]) #1-float(x2.data) #x3[0])
        df.loc[df.image == str(f)[start:], 'bin_score'] = 0 if g == 'low' else 1
df = df.loc[df.pred > -1]
#df['bin_pred'] = df.pred.apply(lambda x: 0 if x < 0.6 else 1)
#df2 = df.groupby('image_name').agg({'pred':np.median, 'bin_score':np.max})
print(roc_auc_score(df.bin_score, df.pred)) # , roc_auc_score(df2.bin_score, df2.pred))
df.to_csv(base_dir+'temp1.csv')
#df2.to_csv(base_dir+'temp2.csv')
'''
def train_model_classification(cyc_len):
        img_dir = Path(base_dir)
        data = ImageDataLoaders.from_folder(img_dir,
                bs=16,
                #valid_pct=0.95,
                train=base_dir+'/train',
                valid=base_dir+'/validate'
                #,item_tfms=Resize(1000),
      #  batch_tfms=aug_transforms(do_flip=True, flip_vert=True, max_rotate=180.0, max_lighting=0.4, p_affine=0.7, p_lighting=0.7, xtra_tfms=Normalize.from_stats(*imagenet_stats))
        ) 

        learn = cnn_learner(data, models.resnet34, metrics=[AUROC()]) #.to_distributed(args.local_rank)

        learn.fit_one_cycle(
                cyc_len=cyc_len, # 30,
                max_lr=5e-3, # 1.3e-6,
                wd=0.4,
                div_factor=10,
                callbacks=[SaveModelCallback(learn, name=f'model-r', monitor='auroc')]) # 3e-3 div_factor=10, max_lr=slice(1e-7, 1e-4), pct_start=0.5
        learn.load(base_dir+f'/models/model-r')

        return learn

def train_model_regression(cyc_len):
    img_dir = Path(base_dir)
    df = pd.read_csv(base_dir+'/data_v2.csv', index_col=0)
    #data_cl = ImageDataBunch.from_folder(img_dir, bs=8,
    #        ds_tfms=get_transforms(do_flip=True, flip_vert=True, max_rotate=180.0, max_lighting=0.4, p_affine=0.7, p_lighting=0.7),
    #        size=512).normalize(imagenet_stats)
    data_re = ImageList.from_folder(img_dir)\
                    .split_by_rand_pct()\
                    .label_from_func(func, label_cls=FloatList)\
                    .transform(get_transforms(do_flip=True, flip_vert=True, max_rotate=180.0, max_lighting=0.4, p_affine=0.7, p_lighting=0.7), size=512)\
                    .databunch(bs=8)\
                    .normalize(imagenet_stats)
    learn = cnn_learner(data_re, models.alexnet, metrics= [rmse,r2_score]) # , metrics=[AUROC()]) #.to_distributed(args.local_rank)
    #learn.load('/home/dsi/zurkin/data20/models/model')
    #learn.loss_func = CrossEntropyFlat(weight=torch.FloatTensor([1.,1.6]).cuda()) #[1.,1.788]).cuda()) # 0.64/0.64, 0.64/0.36.
    #learn.data = data
    #learn.fit(0, lr=0) # Hack for init metric.
    #print(learn.validate(metrics=[AUROC()]))

    #learn.lr_find()
    #fig = learn.recorder.plot(return_fig=True, suggestion=True)
    #fig.savefig('fig')
    #fig = learn.recorder.plot_lr(return_fig=True)
    #fig.savefig('lr')
    #print(learn.recorder.min_grad_lr)

    learn.fit_one_cycle(
                cyc_len=cyc_len, # 30,
                max_lr=5e-4, # 5e-3,
                wd=0.4,
                div_factor=10,
                ) # callbacks=[SaveModelCallback(learn, name=f'model', monitor='auroc')]) # 3e-3 div_factor=10, max_lr=slice(1e-7, 1e-4), pct_start=0.5
    #learn.load(base_dir+f'/models/model')
    return learn

def predict(learn):
        preds = []
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        for g in ['low', 'high']:
                for f in Path(base_dir+"_test/"+g).iterdir():
                        img = open_image(f)
                        for i in range(3):
                                img.data[i,:,:] -= mean[i]
                                img.data[i,:,:] /= std[i]
                        img.resize(torch.Size([img.shape[0],700,700]))
                        x1,x2,x3 = learn.predict(img)
                        #preds.append(int(x))
                        start = str(f).find('TCGA')
                        df.loc[df.image == str(f)[start:], 'pred'] = float(x3[0]) #1-float(x2.data) #x3[0])
                        df.loc[df.image == str(f)[start:], 'bin_score'] = 0 if g == 'low' else 1
        df = df.loc[df.pred > -1]
        df.to_csv('pred_20_test')
        #df['bin_pred'] = df.pred.apply(lambda x: 0 if x < 0.6 else 1)

def use_model(learn):
        base_dir = '/home/dsi/zurkin/data20_test/low/'
        def process_image(x):

                #data = ImageDataBunch.from_folder(Path(base_dir),x,bs=20,size=512).normalize() # imagenet_stats)

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


        df = pd.read_csv('/home/dsi/zurkin/data20_test/temp1.csv')
        df['image_name'] = df.name.apply(lambda x: x[5:12])
        df['bin_score'] = df.score.apply(lambda x: 0 if x < 5.6 else 1)
        # test = pd.read_csv('/home/dsi/zurkin/data20_test/temp1.csv', index_col=0)[['image_name', 'group']]
        # df = pd.merge(df, test, on = 'image_name')
        # df.dropna(inplace=True)
        df2 = df.groupby('image_name').progress_apply(process_image)
        df2.to_csv('/home/dsi/zurkin/20_test.csv')


def test_model_classification(learn):
        data = ImageDataBunch.from_folder(Path(base_dir +'_test/'),
            # bs=2,
            # valid_pct=0.99,
            #train=Path(base_dir+'/train'),
            #valid=Path(base_dir+'/validate'),
            #ds_tfms=get_transforms(do_flip=True, flip_vert=True, max_rotate=180.0, max_lighting=0.2, p_affine=0.4, p_lighting=0.75),
            size=750).normalize() # imagenet_stats)
        learn.data = data
        #learn.fit(0, lr=0) # Hack for init metric.
        print("AUC for patch:",learn.validate(metrics=[AUROC()]))

def test_model(learn):
    data = ImageDataBunch.from_folder(Path(base_dir +'_test/'),
            #bs=2,
            #valid_pct=0.99,
            size=700) #.normalize(imagenet_stats)
    #data = ImageList.from_folder(Path(base_dir+'_test'))\
    #                .split_by_rand_pct(valid_pct=0.99)\
    #                .label_from_func(func, label_cls=FloatList)\
    #                .databunch(no_check=True)\
    #                .normalize(imagenet_stats)
    learn.data = data
    print(learn.validate())

def test_model2_classification(learn):
        df = pd.read_csv('/home/dsi/zurkin/data20_test/temp1.csv', index_col=0)
        df2 = df.groupby('image_name').agg({'pred':np.median, 'bin_score':np.max})
        print("AUC for patch:",roc_auc_score(df.bin_score, df.pred),"AUC for patient:", roc_auc_score(df2.bin_score, df2.pred))


def test_model2(learn=None):
    df = pd.read_csv(base_dir+'/data_v2.csv', index_col=0)
    #test = pd.read_csv(base_dir+'_test/temp1.csv', index_col=0)[['image_name', 'group']]
    #df = pd.merge(df, test, on = 'image_name')
    #df['image_name'] = df.name.apply(lambda x: x[5:12])
    df['bin_score'] = df.score.apply(lambda x: 0 if x < 5.6 else 1)
    df['pred'] = -1
    
    preds = []
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for g in ['low', 'high']:
        for f in Path(base_dir+'_test/'+g).iterdir():
            img = open_image(f)
            for i in range(3):
                img.data[i,:,:] -= mean[i]
                img.data[i,:,:] /= std[i]
            #img.resize(torch.Size([img.shape[0],700,700]))
            x1,x2,x3 = learn.predict(img)
            #preds.append(int(x))
            start = str(f).find('TCGA')
            df.loc[df.image == str(f)[start:], 'pred'] = x1.data[0] # float(x3[0]) #1-float(x2.data) #x3[0])
            df.loc[df.image == str(f)[start:], 'bin_score'] = 0 if g == 'low' else 1
    df = df.loc[df.pred > -1]
    #df['bin_pred'] = df.pred.apply(lambda x: 0 if x < 0.6 else 1)

    #df2 = df.groupby('image_name').agg({'pred':np.median, 'bin_score':np.max})
    df.to_csv(base_dir+'/temp1-r.csv')
    df = pd.read_csv(base_dir+'/temp1-r.csv')
    print(roc_auc_score(df.bin_score, df.pred)) # , roc_auc_score(df2.bin_score, df2.pred))
    r2 = r2_score(df['score'], df['pred']*10)
    print(r2)

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
    df['bin_score'] = df.score.apply(lambda x: 0 if x < 5.6 else 1)
    #test = pd.read_csv('/home/dsi/zurkin/data20_test/temp1.csv', index_col=0)[['image_name', 'group']]
    #df = pd.merge(df, test, on = 'image_name')
    #df.dropna(inplace=True)
    df2 = df.groupby('image_name').progress_apply(process_image)
    df2.to_csv(f'part_b_out.csv')

def train_histograms():
    df = pd.read_csv(base_dir+'/histdata.csv')
    df = df.groupby('image_name').agg({'score':np.max, 'pred':np.median}) #histogram})
    #df['pred'] = df.pred.apply(lambda x: x[0])
    groups = pd.read_csv(base_dir+'/data_v2.csv', index_col=0)[['image_name', 'group']]
    df = pd.merge(df, groups, on = 'image_name')
    data = pd.DataFrame(df.pred.tolist())
    df['label'] = df.score.apply(lambda x: 0 if x < 5.6 else 1)
    train_idx = df.loc[df.group == 'train'].index
    val_idx = df.loc[df.group == 'validate'].index
    test_idx = df.loc[df.group == 'test'].index
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

    pred = df.iloc[test_idx].pred #idxmax(axis=1)
    print(roc_auc_score(df.iloc[test_idx]['label'], pred))



if __name__ == '__main__':
        seed_everything()
        train_model_classification(30)
        # learn = train_model(5)
        # #use_model(learn)
        # test_model2(learn)
        # #train_histograms()

        # df = pd.read_csv(base_dir+'/temp1-r.csv')
        # print(roc_auc_score(df.bin_score, df.pred)) # , roc_auc_score(df2.bin_score, df2.pred))
        # r2 = r2_score(df['score'], df['pred']*10)
        # mse = mean_squared_error(df['score'], df['pred']*10)
        # print("r2: ",r2, "MSE: ",mse)

