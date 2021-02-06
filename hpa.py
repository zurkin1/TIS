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
print(fastai.__version__)


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()


torch.cuda.set_device(5)
path = Path('/home/dsi/zurkin/data/data1/')
df = pd.read_csv(path/'train.csv')
df.columns = ['Id', 'Target']

protein_stats = ([0.08069, 0.05258, 0.05487], [0.13704, 0.10145, 0.15313])
#trn_tfms,_ = get_transforms(do_flip=True, flip_vert=True, max_rotate=30., max_zoom=1, max_lighting=0.05, max_warp=0.)
src = ImageDataLoaders.from_csv(path, csv_fname='train.csv', suff='.png', label_delim='|', bs=64,
                                batch_tfms=aug_transforms(do_flip=True, flip_vert=True, max_rotate=180.0, max_lighting=0.4, p_affine=0.7, p_lighting=0.7, xtra_tfms=Normalize.from_stats(*protein_stats))
      )

f1score_multi = F1ScoreMulti()
def _resnet_split(m): return (m[0][6],m[1])
learn = cnn_learner(
    src,
    alexnet, #resnet18,
    #cut=-2,
    #splitter=_resnet_split,
    loss_func=F.binary_cross_entropy_with_logits,
    path=path,
    metrics=[f1score_multi]
)

#learn.lr_find()
#learn.recorder.plot()

lr = 3e-2
learn.fit_one_cycle(3, slice(lr))
learn.save('stage-1-rn18-datablocks')

learn.unfreeze()
#learn.lr_find()
#learn.recorder.plot()
learn.fit_one_cycle(60, slice(3e-5, lr/5))
learn.save('stage-2-rn18')

#preds,_ = learn.get_preds(DatasetType.Test)
#pred_labels = [' '.join(list([str(i) for i in np.nonzero(row>0.2)[0]])) for row in np.array(preds)]
#df = pd.DataFrame({'Id':test_ids,'Predicted':pred_labels})
#df.to_csv(path/'protein_predictions_datablocks.csv', header=True, index=False)
