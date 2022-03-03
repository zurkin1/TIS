#https://www.kaggle.com/thedrcat/fastai-cell-tile-prototyping-training
#https://www.kaggle.com/anil777/pretrained-resnet34-with-rgby-0-460-public-lb
#https://www.kaggle.com/narainp/hpa-fastai-starter-training
#https://www.kaggle.com/skydevour/rgb-model-rgby-cell-level-classification
#https://www.kaggle.com/thedrcat/cam-class-activation-map-explained-in-pytorch
#https://medium.com/@JBKani/multi-label-classification-of-human-protein-dataset-using-fastai2-86535f96a607
#https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/218309
#https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/228635
#https://www.kaggle.com/lnhtrang/hpa-public-data-download-and-hpacellseg

#!pip install -q timm
#python -m fastai.launch scriptname.py

from fastai.vision.all import *
from fastai.callback.mixup import *
from timm import create_model
import albumentations, timm
from fastai.metrics import accuracy, F1Score
from fastai.losses import BCEWithLogitsLossFlat, FocalLossFlat
import os
from fastai.distributed import *
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch.utils.data import Dataset
from torchvision import transforms
import random
from PIL import Image
import torch
import cv2


root = '/home/dsi/zurkin/data/train/' #train_p/'
nfold = 5
device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else "cpu")
image_res=1024
batch_size = 16
#imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
data_transforms = {
    0: transforms.Compose([
        transforms.RandomResizedCrop(image_res), #224
        transforms.RandomHorizontalFlip(),
        #transforms.ColorJitter(),
        transforms.RandomAffine(90),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    1: transforms.Compose([
        #transforms.RandomResizedCrop(image_res), #256
        transforms.CenterCrop(image_res),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

class HpaDataset(Dataset):
    def __init__(self, is_valid, df):
        self.df = df
        self.df = self.df.loc[self.df.is_valid == is_valid]
        self.df.reset_index(inplace=True, drop=True)
        self.root = root
        self.transform = data_transforms[is_valid]
        self.is_valid = is_valid
        self.vocab = [str(i) for i in range(0, 19)] #https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = np.array(Image.open(f'{self.root}/{self.df.ID[idx]}_green.png'))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        """
        mask = np.load(f'{self.root}public_masks/{self.df.ID[idx]}.npy')
        num_cells = mask.max()
        cell = random.randint(1,num_cells+1)
        mask = np.where(mask==cell,1,0).astype(np.uint8)
        img = img * mask[:,:,None]
        """
        img = Image.fromarray(img)
        img = self.transform(img)
        #label = torch.Tensor([int(self.df.Label.iloc[idx])]).squeeze() #.reshape(-1, 2)
        #label = torch.ByteTensor(int(self.df.Label.iloc[idx]), 19).squeeze()
        label = self.df.Label.iloc[idx].split('|')
        #sample = {'image': image, 'label': label}

        return img, label #.reshape(1,-1) #sample

    def new_empty(self):
        return HpaDataset(False, self.df)


class AlbumentationsTransform(DisplayedTransform):
    split_idx,order=0,2
    def __init__(self, train_aug): store_attr()

    def encodes(self, img: PILImage):
        aug_img = self.train_aug(image=np.array(img))['image']
        return PILImage.create(aug_img)


def get_train_aug(): return albumentations.Compose([
            albumentations.HueSaturationValue(
                hue_shift_limit=0.2,
                sat_shift_limit=0.2,
                val_shift_limit=0.2,
                p=0.5
            ),
            albumentations.CoarseDropout(p=0.5),
            albumentations.RandomContrast(p = 0.6)
])


def get_data():
    files = set([name.rstrip('_green.png') for name in os.listdir(root) if name[0]=='0'])
    df = pd.read_csv(root+'/../public.csv') #train.csv')
    # df = df.sample(frac=0.4).reset_index(drop=True)
    labels = [str(i) for i in range(19)]
    df = df.loc[df.ID.isin(files)]

    for x in labels: df[x] = df.Label.apply(lambda r: int(x in r.split('|')))
    df['fold'] = np.nan
    mskf = MultilabelStratifiedKFold(n_splits=nfold)
    for i, (_, test_index) in enumerate(mskf.split(df['ID'], df[labels])):
        df.iloc[test_index, -1] = i

    df['fold'] = df['fold'].astype('int')
    df['is_valid'] = df.fold.apply(lambda x: x==0)
    #df = df.loc[df.Label.isin(labels)]
    return df

    item_tfms = AlbumentationsTransform(get_train_aug()) #RandomResizedCrop(460, min_scale=0.75, ratio=(1.,1.))
    batch_tfms = [*aug_transforms(size=1024, flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.1), #aug_transforms(do_flip=True, flip_vert=True, max_rotate=180.0, max_lighting=0.4, p_affine=0.7, p_lighting=0.7)
                  Normalize.from_stats([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

    #class PILImageRGBA(PILImage): _show_args, _open_args = {'cmap': 'P'}, {'mode': 'RGBA'}
    cells = DataBlock(blocks=(ImageBlock(PILImageBW), MultiCategoryBlock),
                    get_x=ColReader(0, pref=root, suff='_green.png'),
                    splitter=RandomSplitter(), #splitter=ColSplitter(col='is_valid'),
                    get_y=ColReader(1, label_delim='|'),
                    item_tfms = item_tfms,
                    batch_tfms = batch_tfms)

    dls = cells.dataloaders(df, bs=128)
    #dls.show_batch()
    return dls


def create_timm_body(arch:str, pretrained=True, cut=None):
    model = create_model(arch, pretrained=pretrained)
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    if isinstance(cut, int): return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut): return cut(model)
    else: raise NamedError("cut must be either integer or function")

def get_model():
    body = create_body(resnet50, pretrained=True) #resnext50d_32x4d
    #w = body[0][0].weight
    #body[0][0] = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=2, padding=1, bias=False)
    #body[0][0].weight = nn.Parameter(torch.cat([w, nn.Parameter(torch.mean(w, axis=1).unsqueeze(1))], axis=1))
    nf = num_features_model(nn.Sequential(*body.children())) #* (2)
    head = create_head(nf, 19)
    model = nn.Sequential(body, head)
    apply_init(model[1], nn.init.kaiming_normal_)
    return model.cuda()


if __name__ == '__main__':
    #File check.
    #for file in os.listdir(root):
    #    try:
    #        im = Image.open(root+file)
    #        im.verify()
    #    except:
    #        print(file)
    df = get_data() #dls

    train_ds = HpaDataset(False, df)
    valid_ds = HpaDataset(True, df)
    dls = DataLoaders.from_dsets(train_ds, valid_ds, bs=batch_size, device=device)
    dls.c = 19

    #clweight = compute_class_weight('balanced', classes=range(0,19), y=df.Label.values)
    loss_func = FocalLossFlat(gamma=2) #weight=torch.FloatTensor(clweight).cuda(),
    model = get_model()
    learn = Learner(dls, model, loss_func=loss_func, metrics=[accuracy_multi, PrecisionMulti()], splitter=default_split).to_fp16() #clip=0.5 accuracy_multi, PrecisionMulti()
    #learn = load_learner('baseline')
    #learn.dls = dls
    #learn.freeze()
    #print(learn.lr_find())
    #with learn.distrib_ctx():
    learn.fit_one_cycle(30, 3e-3, cbs=[SaveModelCallback(monitor='precision_score')]) #EarlyStoppingCallback(patience=3),
    learn.export('baseline')
