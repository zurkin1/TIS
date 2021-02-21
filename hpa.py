import os
from pathlib import Path
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2

from fastai.vision.all import *
from fastai.metrics import error_rate
from fastai.distributed import *
from sklearn.metrics import roc_auc_score, mean_squared_error
import argparse
from PIL import Image
from torchvision import transforms
import os
import fastai
import hpacellseg.cellsegmentator as cellsegmentator
from hpacellseg.utils import label_cell
print(fastai.__version__)


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()


torch.cuda.set_device(2)
path = '/home/dsi/zurkin/data/'
protein_stats = ([0.08069, 0.05258, 0.05487], [0.13704, 0.10145, 0.15313])

#trn_tfms,_ = get_transforms(do_flip=True, flip_vert=True, max_rotate=30., max_zoom=1, max_lighting=0.05, max_warp=0.)
src = ImageDataLoaders.from_folder(path+'hpa_train/', valid_pct=0.05, bs=64,
                                batch_tfms=aug_transforms(do_flip=True, flip_vert=True, max_rotate=180.0, max_lighting=0.4, p_affine=0.7, p_lighting=0.7, xtra_tfms=Normalize.from_stats(*protein_stats)))

f1score_multi = F1Score() #F1ScoreMulti()
def _resnet_split(m): return (m[0][6],m[1])
learn = cnn_learner(
    src,
    resnet18,
    #cut=-2,
    #splitter=_resnet_split,
    #loss_func=F.binary_cross_entropy_with_logits,
    path=path+'hpa/',
    #metrics=[f1score_multi]
)

#learn.lr_find()
#learn.recorder.plot()
#learn.load('stage-2-rn18')
#learn = load_learner(path+'hpa/stage-2-rn-18.pkl')
lr = 3e-2
learn.fit_one_cycle(3, slice(lr))
learn.save('stage1-rn18')
learn.unfreeze()
learn.fit_one_cycle(60, slice(3e-5, lr/5))
#learn.save('stage-2-rn18')
learn.export('stage2-rn18-run2.pkl')
