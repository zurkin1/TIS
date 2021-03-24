from fastai.vision.all import *
from fastai.metrics import error_rate
from fastai.distributed import *
from pathlib import Path
from sklearn.metrics import roc_auc_score, mean_squared_error
import argparse
from PIL import Image
from torchvision import transforms
import os


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()


torch.cuda.set_device(7)
#stand = lambda x: (x-5.72)/14.1
stand = lambda x: (x-5.95)/20.44
binarize = lambda x: 0 if x<5.95 else 1
base_dir = '/home/dsi/zurkin/data/skcm' # data24 data23 data31
df = pd.read_csv(base_dir+'/data_v2.csv')
img_dir = Path(base_dir)
#fnames = get_image_files(base_dir)
#labels = [stand(float(x.name.split('_')[0])) for x in fnames]
df['image'] = df.image.apply(lambda x: x[5:12].replace('.','-'))
df['path'] = base_dir + '/' + df.image
df['bin_score'] = df.score.apply(lambda x: binarize(x))
df2 = df.loc[[os.path.isfile(x) for x in df.path.values]].copy()
fnames = df2.path.values.tolist()
labels = [binarize(float(x)) for x in df2.score.values.tolist()]
data_re = ImageDataLoaders.from_lists(img_dir, fnames, labels, bs=4, item_tfms=Resize(500),
        batch_tfms=aug_transforms(do_flip=True, flip_vert=True, max_rotate=180.0, max_lighting=0.4, p_affine=0.7, p_lighting=0.7, xtra_tfms=Normalize.from_stats(*imagenet_stats))
        )
learn = cnn_learner(data_re, resnet34)
#print(learn.lr_find())
learn.fit_one_cycle(60, 2e-5, wd=0.1, cbs=[SaveModelCallback()]) # monitor='roc_auc_score')]) # 3e-3 lr_max=9e-5
#learn.fine_tune(epochs=10, base_lr=2e-5, wd=0.3) # , div_factor=10 , cbs=[SaveModelCallback(monitor='roc_auc_score')]) # 3e-3

# Test model.
df['pred'] = -1
#df['zscore'] = df.score.apply(stand)
#avg_score = np.mean(df.zscore)
#df.zscore.apply(lambda x: 0 if x <= avg_score else 1) # 0.57238 0.56178

#preds = []
#mean = [0.485, 0.456, 0.406]
#std = [0.229, 0.224, 0.225]

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
    #start = str(f).find('TCGA')
    df.loc[df.image == str(f)[-7:], 'pred'] = int(x1[0]) # float(x3[0]) #1-float(x2.data) #x3[0])

df = df.loc[df.pred > -1]
print(roc_auc_score(df.bin_score, df.pred)) # , mean_squared_error(df.zscore, df.pred))

#df['pred'] = avg_score # 0.56178 #Baseline model.
#print(mean_squared_error(df.zscore, df.pred))
