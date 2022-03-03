from itertools import groupby
from pycocotools import mask as mutils
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import pickle
import cv2
from multiprocessing import Pool
import matplotlib.pyplot as plt


conf_name = "mask_rcnn_s101_fpn_syncbn-backbone+head_mstrain_1x_coco"
cell_mask_dir = '/home/zel/dani/data/hpa_full_size/hpa-mask/hpa_cell_mask'
ROOT = '/home/zel/dani/data/hpa_full_size/'
train_or_test = 'train_p1'
img_dir = f'{ROOT}{train_or_test}'
df = pd.read_csv(os.path.join(ROOT, 'train.csv'))
# this script takes more than 9 hours for full data.

# image to run length encoding
def get_rles_from_mask(image_id):
    img = np.load(f'{cell_mask_dir}/{image_id}.npz')['arr_0']
    rle_list = []
    for val in np.unique(img):
        if val == 0:
            continue
        binary_mask = np.where(img == val, val, 0).astype(bool)
        counts = []
        rle = coco_rle_encode(binary_mask)
        rle_list.append(rle)
    return rle_list, img.shape[0], img.shape[1]

def coco_rle_encode(mask):
    rle = {'counts': [], 'size': list(mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

# mmdet custom dataset generator
def mk_mmdet_custom_data(image_id):
    rles, height, width = get_rles_from_mask(image_id)
    if len(rles) == 0:
        return {
            'filename': image_id+'.png',
            'width': width,
            'height': height,
            'ann': {}
        }
    rles = mutils.frPyObjects(rles, height, width)
    masks = mutils.decode(rles)
    bboxes = mutils.toBbox(mutils.encode(np.asfortranarray(masks.astype(np.uint8))))
    bboxes[:, 2] += bboxes[:, 0]
    bboxes[:, 3] += bboxes[:, 1]
    return {
        'filename': image_id+'.png',
        'width': width,
        'height': height,
        'ann':
            {
                'bboxes': np.array(bboxes, dtype=np.float32),
                'labels': np.zeros(len(bboxes)), # dummy data.(will be replaced later)
                'masks': rles
            }
    }


def read_img(image_id):
    filename = f'{ROOT}/{train_or_test}/{image_id}.png'
    assert os.path.exists(filename), f'not found {filename}'
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if img.max() > 255:
        img_max = img.max()
        img = (img/255).astype('uint8')
    return img

# make annotation helper called multi processes
def mk_ann(idx):
    image_id = df.iloc[idx].ID
    anno = mk_mmdet_custom_data(image_id)
    #img = read_img(image_id, train_or_test)
    #cv2.imwrite(f'{img_dir}/{image_id}.jpg', img)
    return anno, idx, image_id


#Generate data for training.
# this part would take several hours, depends on your CPU power.
MAX_THRE = 40 # set your avarable CPU count.
p = Pool(processes=MAX_THRE)
annos = []
len_df = len(df)
for anno, idx, image_id in p.imap(mk_ann, range(len(df))):
    if len(anno['ann']) > 0:
        annos.append(anno)
    #if(idx%100==0):
    print(idx, flush=True)

lbl_cnt_dict = df.set_index('ID').to_dict()['Label']
trn_annos = []
val_annos = []
val_len = int(len(annos)*0.01)
for idx in range(len(annos)):
    ann = annos[idx]
    filename = ann['filename'].replace('.jpg','').replace('.png','')
    label_ids = lbl_cnt_dict[filename]
    len_ann = len(ann['ann']['bboxes'])
    bboxes = ann['ann']['bboxes']
    masks = ann['ann']['masks']
    # asign image level labels to each cells
    for cnt, label_id in enumerate(label_ids.split('|')):
        label_id = int(label_id)
        if cnt == 0:
            ann['ann']['labels'] = np.full(len_ann, label_id)
        else:
            ann['ann']['bboxes'] = np.concatenate([ann['ann']['bboxes'],bboxes])
            ann['ann']['labels'] = np.concatenate([ann['ann']['labels'],np.full(len_ann, label_id)])
            ann['ann']['masks'] = ann['ann']['masks'] + masks    
    if idx < val_len:
        val_annos.append(ann)
    else:
        trn_annos.append(ann)

with open(f'{ROOT}/work/mmdet_full.pkl', 'wb') as f:
    pickle.dump(annos, f)
with open(f'{ROOT}/work/mmdet_trn.pkl', 'wb') as f:
    pickle.dump(trn_annos, f)
with open(f'{ROOT}/work/mmdet_val.pkl', 'wb') as f:
    pickle.dump(val_annos, f)

#Training.
# I just made following config files based on default mask_rcnn.
# The main changes are CustomDataset, num_classes, data path, etc.
# Other than that, I used it as is for mmdetection.
#!ls -l ../mmdetection/configs/hpa_{exp_name}/
config = f'configs/hpa/{conf_name}.py'
# using --no-validate to avoid some errors for custom dataset metrics
additional_conf = '--no-validate --cfg-options'
additional_conf += f' work_dir=../working/work_dir'
additional_conf += f' optimizer.lr=0.0025'
cmd = f'bash -x tools/dist_train.sh {config} 1 {additional_conf}'
print(cmd)
#bash -x tools/dist_train.sh configs/mask_rcnn/mask_rcnn_s50_fpn_syncbn-backbone+head_mstrain_1x_coco.py 1 --no-validate --cfg-options work_dir=/home/dsi/zurkin/data/hpa/work optimizer.lr=0.0025
#python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/train.py configs/mask_rcnn/mask_rcnn_s50_fpn_syncbn-backbone+head_mstrain_1x_coco.py --launcher pytorch --no-validate --cfg-options work_dir=/home/dsi/zurkin/data/hpa/work optimizer.lr=0.0025
#!cd ../mmdetection; {cmd}
#!ls -Rl .
