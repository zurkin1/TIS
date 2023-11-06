from fastai.vision import *
from fastai.metrics import error_rate
from fastai.vision.models.wrn import wrn_22
from fastai.distributed import *
from pathlib import Path
import argparse

# def num_features_model(m:nn.Module)->int:
#     "Return the number of output features for a `model`."
#     for l in reversed(flatten_model(m)):
#         if hasattr(l, 'num_features'): return l.num_features

#python -m torch.distributed.launch --nproc_per_node=4 tis2.py
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank) # 3
torch.distributed.init_process_group(backend='nccl', init_method='env://')

img_dir = Path('/home/dsi/zurkin/data2/')
#csv_file = '/home/dsi/zurkin/set_1-7_only1.csv'
# data = ImageDataBunch.from_csv(p_imgs, csv_labels=csv_file,
#                                fn_col="image", label_col="binaryclass", bs=12, 
#                                ds_tfms=get_transforms(do_flip=True, flip_vert=True,
#                                                       max_rotate=180.0, max_lighting=0.2,
#                                                       p_affine=0.75, p_lighting=0.75)).normalize()

data = ImageDataBunch.from_folder(img_dir, train='train', valid='test', #bs=16,
        ds_tfms=get_transforms(do_flip=True, flip_vert=True, max_rotate=180.0, max_lighting=0.2, p_affine=0.75, p_lighting=0.75)
        ).normalize() #size=500

# body = create_body(arch=models.alexnet, pretrained=True, cut=-2)
# #nf = num_features_model(body) * 2
# head = create_head(nf=512, nc=data.c, ps=0.5)
# print(head)

# model = nn.Sequential(body, head)
#learn = create_cnn_model(models.alexnet, pretrained=True, concat_pool=True, n_in=2, ps=0.5, bn_final=False)
learn = cnn_learner(data, models.alexnet, metrics=[accuracy, AUROC()]).to_distributed(args.local_rank)

#learn.to_fp16()
#learn = load_learner('/home/dsi/zurkin/data2', 'tis.pkl')
learn.load('/home/dsi/zurkin/data2/models/tis')

#learn.lr_find()
#fig = learn.recorder.plot(return_fig=True, suggestion=True)
#fig.savefig('fig')
#print(learn.recorder.min_grad_lr)

learn.fit_one_cycle(2)
learn.unfreeze()
learn.fit_one_cycle(cyc_len=300, max_lr=5*1e-4) # max_lr=5*1e-4) # slice(1e-6,1e-2))
#learn.fit_one_cycle(10, 3e-3, wd=0.4, div_factor=10, pct_start=0.5)
learn.save('tis')

#learn.export('tis.pkl')
learn = load_learner("/home/dsi/zurkin/data2/models/", test=ImageList.from_folder("/home/dsi/zurkin/data2/validate"))
preds = learn.get_preds(ds_type=DatasetType.Test)
print(preds[1])