from fastai.vision.all import *
from fastai.metrics import *
from fastai.distributed import *
from pathlib import Path
from sklearn.metrics import roc_auc_score, mean_squared_error
import argparse
from PIL import Image
from torchvision import transforms
# import torchvision.transforms as T
import os

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()


#python -m torch.distributed.launch --nproc_per_node=4 tis2-rimon.py

# parser = argparse.ArgumentParser()
# parser.add_argument("--local_rank", type=int)
# args = parser.parse_args()
# torch.cuda.set_device(args.local_rank) # 3
# torch.distributed.init_process_group(backend='nccl', init_method='env://')
# torch.cuda.set_device(1)

torch.cuda.set_device(0)
#stand = lambda x: (x-5.72)/14.1
base_dir = "/home/dsi/rimonshy/MS_rimon/data/SKCM/SKCM-polygon_500_40X"
img_dir = Path(base_dir)

data_re = ImageDataLoaders.from_folder(img_dir, train='train', valid='validate', bs=8, item_tfms=Resize(700)#,num_workers=4, pin_memory=True
        ,batch_tfms=aug_transforms(mult=1.0, do_flip=True, flip_vert=True, max_rotate=180.0, max_lighting=0.4, p_affine=0.7, p_lighting=0.7, xtra_tfms=Normalize.from_stats(*imagenet_stats)))

learn = cnn_learner(data_re, models.resnet34, metrics=accuracy)#.to_distributed(args.local_rank)

#print(learn.lr_find())
#learn.fit_one_cycle(n_epoch=10, lr_max=9e-5, wd=0.3, cbs=[SaveModelCallback(monitor='roc_auc_score')]) # 3e-3
#learn.load(base_dir+f'/models/model-1000,e-4,resnet34')

learn.fit_one_cycle(n_epoch=10, # 30,
                lr_max=1e-4, # 1.3e-6,
                wd=0.4,
                #div_factor=10,
                cbs=[SaveModelCallback(monitor='accuracy', fname=f'model-700,e-4,resnet34')]) # 3e-3
                #)
#learn.unfreeze()
#learn.fit_one_cycle(5, cbs=[SaveModelCallback(monitor='accuracy', fname=f'model-resnet34')], lr_max=slice(1e-7,1e-5))

#learn.fine_tune(epochs=2, base_lr=2e-5, wd=0.3, cbs=[SaveModelCallback(monitor='accuracy', fname=f'model-alexnet')]) # , div_factor=10 , cbs=[SaveModelCallback(monitor='roc_auc_score')]) # 3e-3
#learn.save('model')

#learn.load(base_dir+f'/models/model-1000,e-4,resnet34')

# Test nodel.
df = pd.read_csv("/home/dsi/rimonshy/MS_rimon/data/SKCM/SKCM-polygon_500_40X.csv")
#df = pd.read_csv(base_dir+'/data_v1.csv', index_col=0)

df['pred'] = -1
#df['zscore'] = df.score.apply(stand)
med_score = np.median(df.score)
print(med_score)
df['bin_score'] = df.score.apply(lambda x: 0 if x <= med_score else 1) # 0.57238 0.56178

preds = []
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

for a in ['/high/','/low/']:
    print(base_dir+'_test'+a)
    for f in Path(base_dir+'_test'+a).iterdir():
        #img = Image.open(f) # open_image(f)
        #img = cast(array(img), TensorImage)
        #img = transforms.ToTensor()(img) #.unsqueeze_(0)
        #img = transforms.Resize(512)(img).numpy()
        #img = np.transpose(img, (0,1,2)).numpy()
        # for i in range(3):
        #    img[i,:,:] -= mean[i]
        #    img[i,:,:] /= std[i]
        # img.resize(torch.Size([img.shape[0],512,512]))
        x1,x2,x3 = learn.predict(f)
        #print(float(x3[0]))
        #preds.append(x3)
        start = str(f).find('TCGA')
        df.loc[df.image == str(f)[start:], 'pred'] = float(x3[0]) # float(x3[0]) #1-float(x2.data) #x3[0])
        df.loc[df.image == str(f)[start:], 'pred_class'] = x1 # float(x3[0]) #1-float(x2.data) #x3[0])


'''
for f in Path(base_dir+'_test/high').iterdir():
    img_pil = Image.open(f)u
    img_tensor = transforms.ToTensor()(img_pil)
    #img_fastai = Image(img_tensor)
    # img = cast(array(img), TensorImage)
    # img = transforms.ToTensor()(img)#.unsqueeze_(0)
    # img = transforms.Resize(512)(img).numpy()
    # img = np.transpose(img, (0,1,2)).numpy()
    # for i in range(3):
    #    img[i,:,:] -= mean[i]
    #    img[i,:,:] /= std[i]
    # img.resize(torch.Size([img.shape[0],512,512]))
    x1,x2,x3 = learn.predict(img_tensor)
    preds.append(f,int(x1))
    #start = str(f).find('TCGA')
    #df.loc[df.image == str(f)[start:], 'pred'] = x1[0] # float(x3[0]) #1-float(x2.data) #x3[0])
'''

df = df.loc[df.pred != -1]

#df2 = df.groupby('image_name').agg({'pred':np.median, 'bin_score':np.max})
#print(roc_auc_score(df.bin_score, df.pred), mean_squared_error(df.zscore, df.pred)) # , roc_auc_score(df2.bin_score, df2.pred))
#df.to_csv(base_dir+'/temp.csv')
# print(np.corrcoef(df.log2c1qb/10, df.pred))

#df['pred'] = avg_score # 0.56178 #Baseline model.
#print(mean_squared_error(df.zscore, df.pred))
df.to_csv(base_dir+'/prediction-700,e-4,resnet34.csv')

