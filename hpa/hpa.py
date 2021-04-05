import numpy as np
import torchvision
from fastai.vision.all import *
from fastai.metrics import FocalLossFlat
from fastai.callback.mixup import *
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.utils.class_weight import compute_class_weight
import os
import fastai
import argparse


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
#seed_everything()
torch.cuda.set_device(6)

print(fastai.__version__)
path = '/home/dsi/zurkin/data/train'
protein_stats = ([0.07237, 0.04476, 0.07661], [0.13704, 0.10145, 0.1531]) # ([0.08069, 0.05258, 0.05487], [0.13704, 0.10145, 0.15313])
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

#trn_tfms,_ = get_transforms(do_flip=True, flip_vert=True, max_rotate=30., max_zoom=1, max_lighting=0.05, max_warp=0.)
item_tfms = RandomResizedCrop(256, min_scale=0.75, ratio=(1.,1.))
#item_tfms = Resize(128)
#batch_tfms = [*aug_transforms(flip_vert=True, size=128, max_warp=0), Normalize.from_stats(*protein_stats)]
batch_tfms2 = [*aug_transforms(do_flip=True, flip_vert=True, max_rotate=180.0, max_lighting=0.4, p_affine=0.7, p_lighting=0.7), Normalize()] #, xtra_tfms=Normalize.from_stats(*protein_stats)
src = ImageDataLoaders.from_folder(path, valid_pct=0.05, item_tfms=item_tfms, batch_tfms=batch_tfms2) #, bs=64
#f1score_multi = F1Score() #F1ScoreMulti()
#def _resnet_split(m): return (m[0][6],m[1])
#learn = cnn_learner(dls, resnet50, metrics=[accuracy_multi, PrecisionMulti()]).to_fp16()
#df = pd.read_csv(path+'train_single_class.csv')
#clweight = compute_class_weight('balanced', classes=range(0,19), y=df.Label.values)
loss_func = FocalLossFlat(gamma=2) #weight=torch.FloatTensor(clweight).cuda(),
learn = cnn_learner(src, resnet18, path=path, metrics=[accuracy, F1Score(average='weighted')], loss_func=loss_func) #cut=-2, splitter=_resnet_split, loss_func=F.binary_cross_entropy_with_logits, metrics=[f1score_multi, Precision(average='weighted')]

#print(learn.lr_find(suggestions=True))
#learn.recorder.plot()
#learn.load('stage2-rn18')
#learn = load_learner(path+'train/stage2-rn18.pkl')
lr = 5.7e-3

learn.fine_tune(10, base_lr=lr, freeze_epochs=2) #, cbs=[EarlyStoppingCallback(patience=3), SaveModelCallback(monitor='accuracy')])
#learn.fine_tune(2,base_lr=lr)
#learn.fit_one_cycle(3, slice(lr))
#learn.recorder.plot_loss()
#learn.unfreeze()
#learn.fit_one_cycle(30, slice(3e-5, lr/5))
#learn.save('stage2-rn18')
learn.export('rn18-1.pkl')
