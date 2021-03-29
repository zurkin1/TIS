#https://www.kaggle.com/thedrcat/fastai-cell-tile-prototyping-training
#https://www.kaggle.com/anil777/pretrained-resnet34-with-rgby-0-460-public-lb
#https://www.kaggle.com/narainp/hpa-fastai-starter-training
#https://www.kaggle.com/skydevour/rgb-model-rgby-cell-level-classification
#https://www.kaggle.com/thedrcat/cam-class-activation-map-explained-in-pytorch
#https://medium.com/@JBKani/multi-label-classification-of-human-protein-dataset-using-fastai2-86535f96a607

#!pip install -q timm

from fastai.vision.all import *
from fastai.callback.mixup import *
from timm import create_model
import albumentations, timm
from fastai.metrics import accuracy_multi
import os


root = '/home/dsi/zurkin/data/train_p/'


def get_data(df=True):
    if df:
        df = pd.read_csv(root+'../train.csv')
    #df = os.listdir(root)
    #df = pd.DataFrame(df, columns=['ID'])
    #df['Label'] = '0'

    # df = df.sample(frac=0.4).reset_index(drop=True)
    #item_tfms = RandomResizedCrop(460, min_scale=0.75, ratio=(1.,1.))
    batch_tfms = [*aug_transforms(size=320, flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.1), 
                  Normalize.from_stats([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])] #, 0.456 , 0.224

    #class PILImageRGBA(PILImage): _show_args, _open_args = {'cmap': 'P'}, {'mode': 'RGBA'}
    cells = DataBlock(blocks=(ImageBlock(PILImage), MultiCategoryBlock),
                       get_x=ColReader(0, pref=root, suff='.png'),
                       splitter=RandomSplitter(),
                       get_y=ColReader(1, label_delim='|'),
                       #item_tfms = item_tfms,
                       batch_tfms = batch_tfms)

    dls = cells.dataloaders(df)
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
    body = create_timm_body('resnet50', pretrained=True) #resnext50d_32x4d
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
    #learn.freeze()
    #print(learn.lr_find())
    learn.fine_tune(20, base_lr=3e-2, freeze_epochs=1, cbs=[SaveModelCallback(monitor='accuracy_multi')]) #EarlyStoppingCallback(patience=3), 
    learn.export('baseline')