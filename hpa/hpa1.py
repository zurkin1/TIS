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
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from gradcam.utils import visualize_cam
from gradcam import GradCAM


ROOT = '/home/dsi/zurkin/data/test_p/'
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
    red= np.array(Image.open(ROOT+'{}_red.png'.format(img_name)))
    green = np.array(Image.open(ROOT+'{}_{}.png'.format(img_name, greencolor)))
    blue = np.array(Image.open(ROOT+'{}_blue.png'.format(img_name)))
    yellow = np.array(Image.open(ROOT+'{}_yellow.png'.format(img_name)))
    
    #Handle empty images as Cellpose causes an exception.
    if red.sum() < 500:
        red = np.array(Image.open(ROOT+'{}_yellow.png'.format(img_name)))
    img = np.stack((red,green,blue,yellow),-1)
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


class Hook():
    def __init__(self,m):
        self.hook = m.register_forward_hook(self.hook_func)
    def hook_func(self,m,i,o):
        self.stored = o.detach().clone()
    def __enter__(self, *args): return self
    def __exit__(self, *args):
        self.hook.remove()


class PILImageRGBA(PILImage): _show_args, _open_args = {'cmap': 'P'}, {'mode': 'RGBA'}


def visualize_cam1(mask, img, alpha=1.0):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]

    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().detach().numpy()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b]) * alpha

    result = heatmap+img[0][0:3,:,:].cpu()
    result = result.div(result.max()).squeeze()
    Image.fromarray(np.rollaxis(np.uint8(result*255), 0, 3)).save('result.png')

    #return heatmap, result


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
    #LEARN_INT_2_KAGGLE_INT = {k:STR_2_KAGGLE_INT[v] for k,v in LEARN_INT_2_STR.items()}
    LEARN_INT_2_KAGGLE_INT = {x:int(LEARN_LBL_NAMES[x]) for x in np.arange(19)}
    # grayscale=0, R=1, G=2, B=3. channels = [cytoplasm, nucleus]
    channels = [1,3] # red, blue. [[2,3], [0,0], [0,0]]

    for ind, ID in enumerate(ids):
        print(f'Image: {ind} ', ID)
        img = Image.open(ROOT+'{}.png'.format(ID))
        if ID == '15b2d2af-949f-4a6b-afdc-28182fd05212':
            return_dict[ID] = (img.shape, default_rle(img))
            continue
        img_array = np.array(img) #np.transpose(, (2,0,1))
        #Use cellpose for masks. masks (list of 2D arrays, or single 3D array (if do_3D=True)) – labelled image, where 0=no masks; 1,2,…=mask labels.
        mask, flows, styles, diams = model.eval(img_array, diameter=200, channels=channels, do_3D=False, progress=None) #flow_threshold=None,
        if mask.max() == 0:
            return_dict[ID] = (img.shape, default_rle(img))
            continue
        #Get bounding boxes.
        #bboxes = get_contour_bbox_from_raw(mask)
        #if (len(bboxes) == 0):
        #    return_dict[ID] = (img.shape, default_rle(img))
        #    continue
        
        #Cut Out, Pad to Square, and Resize. The first 'cell' in cell_tiles is the whole image and should be ignored.
        #img = read_image(ID, greencolor='green')
        #cell_tiles = [
        #    #cv2.resize(
        #        pad_to_square(img[bbox[1]:bbox[3], bbox[0]:bbox[2], ...])
        #    #   ,TILE_SIZE, interpolation=cv2.INTER_CUBIC) 
        #    for bbox in bboxes]
        
        #Calculate RLEs for all cells ordered by their ID in mask.
        rles = [encode_binary_mask(mask, mask_id) for mask_id in range(mask.max())]
        
        #Get image predictions.
        #('nucleoplasm', tensor(16), tensor([2.0571e-02, 2.7850e-03, 3.8773e-02, 1.0485e-01, 2.2821e-02, 6.9570e-02,...]))
        #for i in range(3):
        #    img[i,:,:] -= imagenet_stats[0][i]
        #    img[i,:,:] /= imagenet_stats[1][i]
        _preds = learn.predict(img_array) #[learn.predict(tile) for tile in cell_tiles]
        torch_img = transforms.Compose([transforms.ToTensor()])(img)[None] # .cuda() transforms.Resize((460, 460)),
        normed_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None] #, 0.456 , 0.224

        #For each class find its explainable cells.
        prediction_str = ""
        all_cell_ids = np.array([])
        for class_idx in (np.where(_preds[2]>CONF_THRESH))[0]:
            #Get explanation.
            #class_idx = LEARN_LBL_NAMES.items.index(clsid)
            clsid = LEARN_LBL_NAMES[class_idx]
            target_layer = learn.model[0]
            gradcam = GradCAM(learn.model, target_layer)
            mask_cam = gradcam(normed_torch_img[0], class_idx=class_idx) #[0] Gradcam mask for one predicted class.
            #visualize_cam1(mask_cam[0], torch_img)
            mask_cam = mask_cam[0].numpy()

            #Select only highly explaining regions.
            explanation_thresh = np.quantile(mask_cam, 0.9)
            mask_cam = np.where(mask_cam>explanation_thresh, 1, 0)

            #Find cells with high explanation. Multiply by Cellpose mask to find relevant cells. Calculate histogram and select only large overlapping regions.
            explained_cells = np.histogram(mask_cam * mask, bins=[i-0.1 for i in range(19)])
            quantile_thresh = np.quantile(explained_cells[0], 0.95) * 0.9
            cell_ids = np.where(explained_cells[0] > quantile_thresh)

            #For each explaining cell build its prediction string.
            for i in cell_ids: #range(1, len(cell_tiles)):
                #classes = np.where(_preds[i][2]>CONF_THRESH)[0]
                #for j in classes:
                prediction_str+=f'{LEARN_INT_2_KAGGLE_INT[int(clsid)]} {_preds[2][class_idx].item()} {rles[i[0]]} '
                all_cell_ids = np.append(all_cell_ids, int(i))

        #For unexplained cells use a negative class.
        for i in set(range(mask.max())) - set(all_cell_ids):
            prediction_str+=f' 18 0.1 {rles[i]}'

        #Save Predictions to Be Added to Dataframe At The End.
        #ImageAID,ImageAWidth,ImageAHeight,class_0 1 rle_encoded_cell_1_mask class_14 1 rle_encoded_cell_1_mask 0 1 rle encoded_cell_2_mask
        return_dict[ID] = (img.shape, prediction_str)


if __name__ == '__main__':
    print(time.ctime())
    manager = Manager()
    return_dict = manager.dict()
    df = []
    model = cellmodels.Cellpose(gpu=True, model_type='cyto') #, device=DEVICE_ID) #, net_avg=False, torch=True
    learn = load_learner('baseline') #'../input/cellpose2/stage2-rn18.pkl')
    num_processes = 1
    X_test = [name.rstrip('.png') for name in (os.listdir(ROOT)) if '.png' in name]
    X_test = pd.DataFrame(X_test, columns=['ID'])
    X = np.array_split(X_test, num_processes)
    print(f'Split length: {len(X[0])}.')
    processes = []
    """
    for rank in range(num_processes):
        p = mp.Process(target=process_image, args=(X[rank],model,learn,return_dict))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    for k,v in return_dict.items():
        #df.loc[df.ID==k,'PredictionString']=v
        df.append([k, v[0][0], v[0][1], v[1]])
    """
    process_image([X_test.ID.iloc[17]],model,learn,return_dict)
    df = pd.DataFrame.from_records(df, columns=['ID', 'ImageWidth', 'ImageHight', 'PredictionString'])
    df.to_csv('/home/dsi/zurkin/data/dataset/submission.csv', index=False)
    print(time.ctime(), len(df))