from fastai.imports import *
from fastai.vision import *
from fastai.metrics import error_rate
import argparse 

img_dir = '/home/dsi/zurkin/data3/'
#csv_file = 'set_1-7_only1.csv'
# path = Path(img_dir)
# dest = path/"folder"
# dest.mkdir(parents=True, exist_ok=True)

path_hr = Path('/home/dsi/zurkin/data3/')
​path_l = path_hr/'small-500'
path_s = path_hr/'small-250'
il = ImageList.from_folder(path_hr)
​sets = [(path_l, 500), (path_s, 250)]


def resize_one(fn, i, path, size):
    dest = path/fn.relative_to(path_hr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = PIL.Image.open(fn)
    targ_sz = resize_to(img, size, use_min=True)
    img = img.resize(targ_sz, resample=PIL.Image.BILINEAR).convert('RGB')
    img.save(dest, quality=75)

def train():

    # data = ImageDataBunch.from_csv(img_dir, csv_labels=csv_file, valid_pct=0.2, fn_col="image", label_col="binaryclass",
    #                            ds_tfms=get_transforms(do_flip=True, flip_vert=True,
    #                                                   max_rotate=10.0, max_lighting=0.2,
    #                                                   p_affine=0.75, p_lighting=0.75)).normalize()

    data = ImageDataBunch.from_folder(img_dir, train='train', valid='test', ds_tfms=get_transforms(do_flip=True, flip_vert=True,
                                                      max_rotate=10.0, max_lighting=0.2,
                                                      p_affine=0.75, p_lighting=0.75),size=500)
    data.normalize(imagenet_stats)
    print(data.classes)
    learn = cnn_learner(data, models.resnet34, metrics=[accuracy, AUROC()])                                  
    #learn.model = nn.DataParallel(learn.model, device_ids=[1, 3, 4, 5])  
    learn.fit_one_cycle(10, max_lr=slice(1e-7,1e-6))
    learn.purge()

    learn.unfreeze
    learn.fit_one_cycle(10, max_lr=slice(1e-7,1e-6))
    learn.save('stage-1')

# def predict(model_name,pred_dir):
#     data = ImageDataBunch.from_folder(model_name, ds_tfms=get_transforms(), size=500)#.normalize(imagenet_stats)
#     learn = create_cnn(data, models.resnet34, metrics=[accuracy, AUROC()])
#     learn = learn.load(model_name)
#     preds = []
#     for f in Path(pred_dir).iterdir():
#         _,x,_ = learn.predict(open_image(f))
#         preds.append(int(x))
#     return preds
for p,size in sets:
    if not p.exists(): 
        print(f"resizing to {size} into {p}")
        parallel(partial(resize_one, path=p, size=size), il.items)

#train()
# preds = predict('/home/dsi/zurkin/data2/models/','/home/dsi/zurkin/data2/validate/')
# print(len(preds),preds)