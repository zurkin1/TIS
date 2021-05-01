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
from fastai.metrics import accuracy_multi
import os
from fastai.distributed import *
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


root = '/home/dsi/zurkin/data/public_masked/' #train_p/'
nfold = 5


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
    files = set([name.rstrip('.png') for name in os.listdir(root)])
    df = pd.read_csv(root+'../public.csv') #train.csv')
    df = df.loc[df.ID.isin(files)]
    #df = os.listdir(root)
    #df = pd.DataFrame(df, columns=['ID'])
    #df['Label'] = '0'
    # df = df.sample(frac=0.4).reset_index(drop=True)
    labels = [str(i) for i in range(19)]
    for x in labels: df[x] = df.Label.apply(lambda r: int(x in r.split('|')))

    df['fold'] = np.nan
    mskf = MultilabelStratifiedKFold(n_splits=nfold)
    for i, (_, test_index) in enumerate(mskf.split(df['ID'], df[labels])):
        df.iloc[test_index, -1] = i

    df['fold'] = df['fold'].astype('int')
    df['is_valid'] = df.fold.apply(lambda x: x==0)

    item_tfms = AlbumentationsTransform(get_train_aug()) #RandomResizedCrop(460, min_scale=0.75, ratio=(1.,1.))
    batch_tfms = [*aug_transforms(size=320, flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.1),
                  Normalize.from_stats([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])] #, 0.456 , 0.224

    #class PILImageRGBA(PILImage): _show_args, _open_args = {'cmap': 'P'}, {'mode': 'RGBA'}
    cells = DataBlock(blocks=(ImageBlock(PILImage), MultiCategoryBlock),
                    get_x=ColReader(0, pref=root, suff='.png'),
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

def get_model(dls):
    body = create_body(resnet50, pretrained=True) #resnext50d_32x4d
    #w = body[0][0].weight
    #body[0][0] = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=2, padding=1, bias=False)
    #body[0][0].weight = nn.Parameter(torch.cat([w, nn.Parameter(torch.mean(w, axis=1).unsqueeze(1))], axis=1))
    nf = num_features_model(nn.Sequential(*body.children())) #* (2)
    head = create_head(nf, dls.c)
    model = nn.Sequential(body, head)
    apply_init(model[1], nn.init.kaiming_normal_)
    return model


if __name__ == '__main__':
    #File check.
    #for file in os.listdir(root):
    #    try:
    #        im = Image.open(root+file)
    #        im.verify()
    #    except:
    #        print(file)
    dls = get_data()
    learn = Learner(dls, get_model(dls), loss_func=BCEWithLogitsLossFlat(), metrics=[accuracy_multi, PrecisionMulti()], splitter=default_split).to_fp16() #clip=0.5
    #learn = load_learner('baseline')
    #learn.dls = dls
    #learn.freeze()
    #print(learn.lr_find())
    #with learn.distrib_ctx():
    learn.fit_one_cycle(30, 3e-4, cbs=[SaveModelCallback(monitor='precision_score')]) #EarlyStoppingCallback(patience=3),
    learn.export('baseline')