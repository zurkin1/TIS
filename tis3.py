from fastai import *
from fastai.vision import *
from fastai.metrics import error_rate
from fastai.vision.models.wrn import wrn_22
from fastai.distributed import *
from pathlib import Path
import argparse
from torch import *
def _default_split(m:nn.Module): return (m[1],)
def _resnet_split(m:nn.Module): return (m[0][6],m[1])
_default_meta = {'cut':-1, 'split':_default_split}
_resnet_meta  = {'cut':-2, 'split':_resnet_split }
model_meta = {
    models.resnet18 :{**_resnet_meta}, models.resnet34: {**_resnet_meta},
    models.resnet50 :{**_resnet_meta}, models.resnet101:{**_resnet_meta},
    models.resnet152:{**_resnet_meta}}


def cnn_config(arch):
    torch.backends.cudnn.benchmark = True
    return model_meta.get(arch, _default_meta)

def num_features_model(m:nn.Module)->int:
    "Return the number of output features for a `model`."
    for l in reversed(flatten_model(m)):
        if hasattr(l, 'num_features'): return l.num_features

#flatten_model = lambda m: sum(map(flatten_model,m.children()),[]) if num_children(m) else [m]

def create_body(model:nn.Module, cut:Optional[int]=None, body_fn:Callable[[nn.Module],nn.Module]=None):
    "Cut off the body of a typically pretrained `model` at `cut` or as specified by `body_fn`."
    return (nn.Sequential(*list(model.children())[:cut]) if cut
            else body_fn(model) if body_fn else model)

def create_head(nf:int, nc:int, lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5):
    """Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes.
    :param ps: dropout, can be a single float or a list for each layer."""
    lin_ftrs = [nf, 512, nc] if lin_ftrs is None else [nf] + lin_ftrs + [nc]
    ps = listify(ps)
    if len(ps)==1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    layers = [AdaptiveConcatPool2d(), Flatten()]
    for ni,no,p,actn in zip(lin_ftrs[:-1],lin_ftrs[1:],ps,actns):
        layers += bn_drop_lin(ni,no,True,p,actn)
    return nn.Sequential(*layers)

def split(self, split_on:SplitFuncOrIdxList)->None:
        "Split the model at `split_on`."
        if isinstance(split_on,Callable): split_on = split_on(self.model)
        self.layer_groups = split_model(self.model, split_on)

# def create_cnn(data:DataBunch, arch:Callable, cut:Union[int,Callable]=None, pretrained:bool=True,
#                 lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5,
#                 custom_head:Optional[nn.Module]=None, split_on:Optional[SplitFuncOrIdxList]=None,
#                 classification:bool=True, **kwargs:Any):
#     "Build convnet style learners."
#     meta = cnn_config(arch)
#     body = create_body(arch(pretrained))
#     nf = num_features_model(body) * 2
#     head = custom_head or create_head(nf, data.c, lin_ftrs, ps)
#     model = nn.Sequential(body, head)
#     learn = Learner(data, model, **kwargs)
#     learn.split(ifnone(split_on,meta['split']))
#     if pretrained: learn.freeze()
#     apply_init(model[1], nn.init.kaiming_normal_)
#     return learn


def create_cnn(data:DataBunch, arch:Callable, cut:Union[int,Callable]=None, pretrained:bool=True,
                lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5,
                custom_head:Optional[nn.Module]=None, split_on:Optional[SplitFuncOrIdxList]=None,
                classification:bool=True, **kwargs:Any)->None:
    meta = cnn_config(arch)
    body = create_body(arch(pretrained), ifnone(cut,meta['cut']))
    nf = num_features_model(body) * 2
    head = custom_head or create_head(nf, data.c, lin_ftrs, ps)
    model = nn.Sequential(body, head)
    learn = Learner(data, model, **kwargs)
    learn.split(ifnone(split_on,meta['split']))
    if pretrained: learn.freeze()
    apply_init(model[1], nn.init.kaiming_normal_)
    return learn
####################################################################################
#python -m torch.distributed.launch --nproc_per_node=4 tis2.py
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank) # 3
torch.distributed.init_process_group(backend='nccl', init_method='env://')

base_dir = '/home/dsi/zurkin/data4_new/'
img_dir = Path(base_dir)

data = ImageDataBunch.from_folder(img_dir, train='train', valid='test', #bs=16,
        ds_tfms=get_transforms(do_flip=True, flip_vert=True, max_rotate=180.0, max_lighting=0.2, p_affine=0.75, p_lighting=0.75)
       ,size=500).normalize() #size=500

learn = cnn_learner(data, models.alexnet, metrics=[AUROC()]) #.to_distributed(args.local_rank)
#learn = load_learner('/home/dsi/zurkin/data2', 'tis.pkl')
# learn.load(base_dir + '/models/bestmodel')
# learn.to_fp16()

# learn.load('/home/dsi/zurkin/data2/models/tis')
learn.fit_one_cycle(2)
learn.unfreeze()
learn.fit_one_cycle(cyc_len=10, max_lr=5*1e-4) # max_lr=5*1e-4) # slice(1e-6,1e-2))

learn.save('tis')
learn.export('tis.pkl')

