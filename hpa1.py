import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

from cellpose import models as cellmodels
from cellpose import plot
from fastai.vision.all import *
import pandas as pd
import numpy as np
from tqdm import tqdm
import imageio
from PIL import Image
import base64
from pycocotools import _mask as coco_mask
import typing as t
import zlib
import time
import os
import io
import matplotlib.pyplot as plt
import cv2
import torch.multiprocessing as mp
from torch.multiprocessing import Manager


ROOT = '/home/dsi/zurkin/data/'
torch.multiprocessing.set_start_method('spawn', force=True)


def encode_binary_mask(mask, mask_id=1): #contour, image_shape):
  """Converts a binary mask into OID challenge encoding ascii text."""
  mask = np.where(mask==mask_id, 1, 0).astype(np.bool)
  # check input mask --
  if mask.dtype != np.bool:
      raise ValueError(f"encode_binary_mask expects a binary mask, received dtype == {mask.dtype}")

  mask = np.squeeze(mask)
  assert len(mask.shape) == 2

  # convert input mask to expected COCO API input --
  mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
  mask_to_encode = mask_to_encode.astype(np.uint8)
  mask_to_encode = np.asfortranarray(mask_to_encode)

  # RLE encode mask --
  encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

  # compress and base64 encoding --
  binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
  base64_str = base64.b64encode(binary_str)
  return base64_str.decode() #'ascii'


def read_image(img_name, greencolor='green'):
    global ROOT
    #green = cv2.imread(ROOT+'/test/{}_{}.png'.format(img_name, greencolor), cv2.IMREAD_GRAYSCALE)
    #red = cv2.imread(ROOT+'/test/{}_red.png'.format(img_name), cv2.IMREAD_GRAYSCALE)
    #blue = cv2.imread(ROOT+'/test/{}_blue.png'.format(img_name), cv2.IMREAD_GRAYSCALE)
    red= np.array(Image.open(ROOT+'/test/{}_red.png'.format(img_name)))
    green = np.array(Image.open(ROOT+'/test/{}_{}.png'.format(img_name, greencolor)))
    blue = np.array(Image.open(ROOT+'/test/{}_blue.png'.format(img_name)))
    
    #Handle empty images as Cellpose causes an exception.
    if red.sum() < 500:
        red = blue
    img = np.stack((red,green,blue),-1)
    return img.astype(np.uint8)
    


def get_contour_bbox_from_raw(mask):
    """ Get bbox of contour as `xmin ymin xmax ymax`
        raw_mask (nparray): Numpy array containing segmentation mask information
    Returns:
        Numpy array for a cell bounding box coordinates
    """
    xy = []
    
    for i in range(0,mask.max()):
        cell_mask = np.where(mask==i, 1, 0).astype(np.uint8)
        cnts, hierarchy = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #RETR_FLOODFILL
        #cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > 100000]
        xywh = cv2.boundingRect(cnts[0]) #x,y,width,hight.
        xy_cell = (xywh[0], xywh[1], xywh[0]+xywh[2], xywh[1]+xywh[3])
        xy.append(xy_cell)
    return xy


def pad_to_square(a):
    """ Pad an array `a` evenly until it is a square """
    if a.shape[1]>a.shape[0]: # pad height
        n_to_add = a.shape[1]-a.shape[0]
        top_pad = n_to_add//2
        bottom_pad = n_to_add-top_pad
        a = np.pad(a, [(top_pad, bottom_pad), (0, 0), (0, 0)], mode='constant')

    elif a.shape[0]>a.shape[1]: # pad width
        n_to_add = a.shape[0]-a.shape[1]
        left_pad = n_to_add//2
        right_pad = n_to_add-left_pad
        a = np.pad(a, [(0, 0), (left_pad, right_pad), (0, 0)], mode='constant')
    else:
        pass
    return a


def default_rle(img):
    if img.shape[0] == 2048:
        sp = '0 0.1 eNoLCAgIMAEABJkBdQ=='
    elif img.shape[0] == 1728:
        sp = '0 0.1 eNoLCAjJNgIABNkBkg=='
    else:
        sp = '0 0.1 eNoLCAgIsAQABJ4Beg=='
    return sp


def process_image(ids, model, learn, return_dict):
    TILE_SIZE = (256,256)
    CONF_THRESH = 0.1

    use_GPU = cellmodels.use_gpu()
    print('>>> GPU activated? %d'%use_GPU)
    LEARN_LBL_NAMES = learn.dls.vocab
    KAGGLE_LBL_NAMES = ["nucleoplasm", "nuclear_membrane", "nucleoli", "nucleoli_fibrillar_center", "nuclear_speckles",\
                        "nuclear_bodies", "endoplasmic_reticulum", "golgi_apparatus", "intermediate_filaments", "actin_filaments",\
                        "microtubules", "mitotic_spindle", "centrosome", "plasma_membrane", "mitochondria", "aggresome", "cytosol",\
                        "vesicles", "negative"]
    LEARN_INT_2_STR = {x:LEARN_LBL_NAMES[x] for x in np.arange(19)}
    KAGGLE_INT_2_STR = {x:KAGGLE_LBL_NAMES[x] for x in np.arange(19)}
    STR_2_KAGGLE_INT = {v:k for k,v in KAGGLE_INT_2_STR.items()}
    LEARN_INT_2_KAGGLE_INT = {k:STR_2_KAGGLE_INT[v] for k,v in LEARN_INT_2_STR.items()}
    # grayscale=0, R=1, G=2, B=3. channels = [cytoplasm, nucleus]
    channels = [1,3] # red, blue. [[2,3], [0,0], [0,0]]


    for ind, ID in enumerate(ids):
        print(f'Image: {ind}')
        img = read_image(ID)
        #Use cellpose for masks. masks (list of 2D arrays, or single 3D array (if do_3D=True)) – labelled image, where 0=no masks; 1,2,…=mask labels.
        mask, flows, styles, diams = model.eval(img, diameter=200, channels=channels, do_3D=False, progress=None) #flow_threshold=None,
        if mask.max() == 0:
            return_dict[ID] = default_rle(img)
            continue
        #Get bounding boxes.
        bboxes = get_contour_bbox_from_raw(mask)
        if (len(bboxes) == 0):
            return_dict[ID] = default_rle(img)
            continue
        
        #Cut Out, Pad to Square, and Resize. The first 'cell' in cell_tiles is the whole image and should be ignored.
        #img = read_image(ID, greencolor='green')
        cell_tiles = [
            #cv2.resize(
                pad_to_square(img[bbox[1]:bbox[3], bbox[0]:bbox[2], ...])
            #   ,TILE_SIZE, interpolation=cv2.INTER_CUBIC) 
            for bbox in bboxes]
        
        #Calculate RLEs for all cells ordered by their ID in mask.
        rles = [encode_binary_mask(mask, mask_id) for mask_id in range(mask.max())]
        
        #Get slide predictions.
        #('nucleoplasm', tensor(16), tensor([2.0571e-02, 2.7850e-03, 3.8773e-02, 1.0485e-01, 2.2821e-02, 6.9570e-02,...]))
        _preds = [learn.predict(tile) for tile in cell_tiles]
        
        #Post-Process: keep only highly confidence classes.
        prediction_str = ""
        for i in range(1, len(cell_tiles)):
            classes = np.where(_preds[i][2]>CONF_THRESH)[0]
            for j in classes:
                prediction_str+=f'{LEARN_INT_2_KAGGLE_INT[j]} {_preds[i][2][j].item()} {rles[i]} '
        
        #Save Predictions to Be Added to Dataframe At The End.
        #ImageAID,ImageAWidth,ImageAHeight,class_0 1 rle_encoded_cell_1_mask class_14 1 rle_encoded_cell_1_mask 0 1 rle encoded_cell_2_mask
        return_dict[ID] = prediction_str


if __name__ == '__main__':
    print(time.ctime())
    manager = Manager()
    return_dict = manager.dict()
    df = pd.read_csv(ROOT+'sample_.csv')
    model = cellmodels.Cellpose(gpu=True, model_type='cyto') #, device=DEVICE_ID) #, net_avg=False, torch=True
    learn = load_learner(ROOT+'train/rn50-1.pkl') #'../input/cellpose2/stage2-rn18.pkl')
    num_processes = 6
    X_test = [name.rstrip('green.png').rstrip('_') for name in (os.listdir(ROOT+'/test/')) if '_green.png' in name]
    X = np.array_split(X_test, num_processes)
    print(f'Split length: {len(X[0])}.')
    processes = []

    for rank in range(num_processes):
        p = mp.Process(target=process_image, args=(X[rank],model,learn,return_dict))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    for k,v in return_dict.items():
        df.loc[df.ID==k,'PredictionString']=v

    #df = df.parallel_apply(process_image, axis=1)
    df.to_csv('/home/dsi/zurkin/data/dataset/submission.csv', index=False)
    print(time.ctime(), len(df))