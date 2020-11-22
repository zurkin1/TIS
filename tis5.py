from fastai.vision.all import *
from fastai.metrics import error_rate
from fastai.distributed import *
from pathlib import Path
from sklearn.metrics import roc_auc_score, mean_squared_error
import argparse
from PIL import Image
from torchvision import transforms


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()


torch.cuda.set_device(0) #args.local_rank) # 3
stand = lambda x: (x-5.72)/14.1
func1 = lambda x: stand(float(x.parts[-1].split('_')[0])) # Standardization.
def func(x):
    val = stand(float(x.split('_')[0]))
    return val
base_dir = '/home/dsi/zurkin/data/data23'
img_dir = Path(base_dir)
fnames = get_image_files(base_dir)
labels = [stand(float(x.name.split('_')[0])) for x in fnames]
data_re = ImageDataLoaders.from_lists(img_dir, fnames, labels, bs=8, item_tfms=Resize(512), batch_tfms=aug_transforms(size=512)) #size=500
#data_re = ImageList.from_folder(img_dir)\
#                .split_by_rand_pct()\
#                .label_from_func(func, label_cls=FloatList)\
#                .transform(get_transforms(do_flip=True, flip_vert=True, max_rotate=180.0, max_lighting=0.4, p_affine=0.7, p_lighting=0.7), size=512)\
#                .databunch(bs=8)\
#                .normalize(imagenet_stats)
learn = cnn_learner(data_re, resnet34) # , metrics=[RocAucBinary()]) #.to_distributed(args.local_rank)
#learn.load(base_dir+'models/model')
learn.fit_one_cycle(n_epoch=30, lr_max=9e-4, wd=0.3) # , div_factor=10, cbs=[SaveModelCallback(monitor='roc_auc_score')]) # 3e-3 

# Test nodel.
df = pd.read_csv(base_dir+'/data_v2.csv', index_col=0)
df['pred'] = -1
df['zscore'] = df.score.apply(stand)
avg_score = np.mean(df.zscore)
df['bin_score'] = df.zscore.apply(lambda x: 0 if x <= avg_score else 1) # 0.57238 0.56178

preds = []
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

for f in Path(base_dir+'_test/').iterdir():
    #img = Image.open(f) # open_image(f)
    #img = cast(array(img), TensorImage)
    #img = transforms.ToTensor()(img) #.unsqueeze_(0)
    #img = transforms.Resize(512)(img).numpy()
    #img = np.transpose(img, (0,1,2)).numpy()
    #for i in range(3):
    #    img[i,:,:] -= mean[i]
    #    img[i,:,:] /= std[i]
    #img.resize(torch.Size([img.shape[0],512,512]))
    x1,x2,x3 = learn.predict(f)
    #preds.append(int(x))
    start = str(f).find('TCGA')
    df.loc[df.image == str(f)[start:], 'pred'] = x1[0] # float(x3[0]) #1-float(x2.data) #x3[0])

df = df.loc[df.pred > -1]
#df2 = df.groupby('image_name').agg({'pred':np.median, 'bin_score':np.max})
print(roc_auc_score(df.bin_score, df.pred), mean_squared_error(df.zscore, df.pred)) # , roc_auc_score(df2.bin_score, df2.pred))
#df.to_csv(base_dir+'/temp.csv')
df['pred'] = avg_score # 0.56178 #Baseline model.
print(mean_squared_error(df.zscore, df.pred))
#df2.to_csv(base_dir+'data_v3d.csv')
