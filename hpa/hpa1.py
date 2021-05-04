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
from cellpose import io
import matplotlib.pyplot as plt
import cv2
import torch.multiprocessing as mp
from torch.multiprocessing import Manager
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from gradcam.utils import visualize_cam
from gradcam import GradCAM
import sys
#from puzzle_utils import *
#from puzzle_networks import *
from lime import lime_image
import albumentations
from hpa import AlbumentationsTransform, HpaDataset


ROOT = '/home/dsi/zurkin/data/'
torch.multiprocessing.set_start_method('spawn', force=True)
TILE_SIZE = (256,256)
CONF_THRESH = 0.5
learn = load_learner('baseline') #'../input/cellpose2/stage2-rn18.pkl')
load_model('./models/model.pth', model=learn, opt="", with_opt=False)
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


def encode_binary_mask(mask, mask_id): #contour, image_shape):
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
    yellow = Image.open(ROOT+'test/{}_yellow.png'.format(img_name))
    """
    red= np.array(Image.open(ROOT+'{}_red.png'.format(img_name)).resize((512,512)))
    green = np.array(Image.open(ROOT+'{}_{}.png'.format(img_name, greencolor)).resize((512,512)))
    blue = np.array(Image.open(ROOT+'{}_blue.png'.format(img_name)).resize((512,512)))
    shape = yellow.shape
    #Handle empty images as Cellpose causes an exception.
    if red.sum() < 500 or img_name == '15b2d2af-949f-4a6b-afdc-28182fd05212':
        red = np.array(yellow.resize((512,512)))
    img = np.stack((red,green,blue),-1)
    if img.max()>257:
        img = img/256
    """
    img = np.array(Image.open(ROOT+f'test_p/{img_name}.png'))
    return img.astype(np.uint8), yellow.shape



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


def default_rle(shape):
    if shape[0] == 2048:
        sp = '0 0.1 eNoLCAgIMAEABJkBdQ=='
    elif shape[0] == 1728:
        sp = '0 0.1 eNoLCAjJNgIABNkBkg=='
    else:
        sp = '0 0.1 eNoLCAgIsAQABJ4Beg=='
    return sp


def visualize_cam1(mask, img, ID, clsid):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]

    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    alpha=1.0
    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().detach().numpy()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b]) * alpha

    result = heatmap+img[0][0:3,:,:].cpu()
    result = result.div(result.max()).squeeze()
    Image.fromarray(np.rollaxis(np.uint8(result*255), 0, 3)).save(ROOT+f'test_p3/{ID}_{clsid}.png')

    #return heatmap, result


def prepare_data(ids, model):
    channels = [1,3]
    for ind, ID in enumerate(tqdm(ids)):
        ids_processed = [x.split('.')[0] for x in os.listdir(ROOT+'public_masks/')]
        if ID in ids_processed:
            continue
        print(ID, ' processed.')
        img = np.array(Image.open(ROOT+f'public/{ID}.png'))
        #img_array = np.array(img) #np.transpose(, (2,0,1))
        #Use cellpose for masks. masks (list of 2D arrays, or single 3D array (if do_3D=True)) – labelled image, where 0=no masks; 1,2,…=mask labels.
        try:
            mask, flows, styles, diams = model.eval(img, diameter=200, channels=channels, do_3D=False, progress=None) #flow_threshold=None,
            if mask.max() <= 4:
                mask, flows, styles, diams = model.eval(img, diameter=100, channels=channels, do_3D=False, progress=None)
            #mask = np.where(mask>0, 1, 0)
            #img = img * mask[:, :, None]
            #Image.fromarray(np.uint8(img)).save(ROOT+'public_masked/'+ID+'.png')
            np.save(ROOT+'public_masks/'+ID+'.npy', mask)
        except:
            print(f'Failed to process image {ID}')


#Image rescaling to (nR * nC) size.
def scale(im, nR, nC):
    nR0 = len(im)     # source number of rows
    nC0 = len(im[0])  # source number of columns
    return np.array([[ im[int(nR0 * r / nR)][int(nC0 * c / nC)]
             for c in range(nC)] for r in range(nR)])


def process_image(ids, model, learn):
    for ind, ID in enumerate(tqdm(ids)):
        ids_processed = [x.split('.')[0] for x in os.listdir(ROOT+'test_p2/')]
        if ID in ids_processed:
            continue
        print(f'Processing {ID}.')
        img, orig_shape = read_image(ID)
        #img_array = np.array(img) #np.transpose(, (2,0,1))
        #Use cellpose for masks. masks (list of 2D arrays, or single 3D array (if do_3D=True)) – labelled image, where 0=no masks; 1,2,…=mask labels.
        #mask, flows, styles, diams = model.eval(img, diameter=200, channels=channels, do_3D=False, progress=None) #flow_threshold=None,
        #if mask.max() <= 4:
        #    mask, flows, styles, diams = model.eval(img, diameter=100, channels=channels, do_3D=False, progress=None)
        #io.save_masks(img, mask, flows, ROOT+f'test_p3/{ID}.png')
        mask = np.load(ROOT+'test_masks/'+ID+'.npy')
        mask_bin = np.where(mask > 0, 1, 0)
        img = np.uint8(img*mask_bin[:, :, None])

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
        orig_mask = scale(mask, orig_shape[0], orig_shape[1])
        rles = [encode_binary_mask(orig_mask, mask_id) for mask_id in range(mask.max()+1)]

        #Get image predictions.
        #('nucleoplasm', tensor(16), tensor([2.0571e-02, 2.7850e-03, 3.8773e-02, 1.0485e-01, 2.2821e-02, 6.9570e-02,...]))
        #for i in range(3):
        #    img[i,:,:] -= imagenet_stats[0][i]
        #    img[i,:,:] /= imagenet_stats[1][
        _preds = learn.predict(img) #[learn.predict(tile) for tile in cell_tiles]
        torch_img = transforms.Compose([transforms.ToTensor()])(img)[None] # .cuda() transforms.Resize((460, 460)),
        normed_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None] #, 0.456 , 0.224

        #Get explanation. For each class find its explainable cells.
        prediction_str = ""
        all_cells = {}

        #Gradcam.
        class_idxs = np.where(_preds[2]>CONF_THRESH)[0]
        #class_idxs = np.argpartition(_preds[2], -4)[-4:].numpy()
        target_layer = learn.model[0]
        gradcam = GradCAM(learn.model, target_layer)

        #Lime.
        #classifier_fn = lambda x: [learn.predict(i)[2].numpy() for i in x]
        #explainer = lime_image.LimeImageExplainer()
        #explanation = explainer.explain_instance(img, classifier_fn, top_labels=4, num_samples=190) #hide_color=0,
        #class_idxs = explanation.top_labels

        for class_idx in class_idxs:
            mask_cam = gradcam(normed_torch_img[0], class_idx=class_idx) #[0] Gradcam mask for one predicted class.
            visualize_cam1(mask_cam[0], torch_img, ID, class_idx)
            mask_cam = mask_cam[0].numpy()

            #img_cpy, mask_cam = explanation.get_image_and_mask(class_idx, positive_only=True)

            #Find cells with high explanation. Multiply by Cellpose mask to find relevant cells. Calculate histogram and select only large overlapping regions.
            mask_cam_bin = np.where(mask_cam>0, 1, 0)
            explained_cells = np.histogram(mask_cam_bin * mask, bins=range(mask.max()+2))
            _thresh = 0.5 * mask_bin.sum()/mask.max() #0.5 * img.shape[0]**2/mask.max() #Approximated cell area in pixels.
            cell_ids = np.where(explained_cells[0] > _thresh)

            #For each explaining cell build its prediction string.
            for i in np.delete(cell_ids, 0): #range(1, len(cell_tiles)):
                #classes = np.where(_preds[i][2]>CONF_THRESH)[0]
                #for j in classes:
                mask_bin = np.where(mask==i, 1, 0)
                iou = (mask_cam_bin * mask_bin).sum()
                #avg_cam = iou/mask_bin.sum()
                avg_cam = np.mean(mask_cam * mask_bin)
                cell_pred = avg_cam*_preds[2][class_idx].item()
                clsid = LEARN_LBL_NAMES[class_idx]
                if (i not in all_cells) or all_cells[i]*10 < cell_pred:
                    all_cells[i] = cell_pred
                    prediction_str+=f'{int(clsid)} {cell_pred} {rles[i]} ' #LEARN_INT_2_KAGGLE_INT[

        #For unexplained cells use a random class.
        for i in set(range(mask.max()+1)) - set(all_cells.keys()) - {0}:
            class_idx = np.random.choice(class_idxs, 1)[0]
            clsid = LEARN_LBL_NAMES[class_idx]
            prediction_str+=f'{int(clsid)} {avg_cam*_preds[2][class_idx].item()} {rles[i]} '

        #Save Predictions to Be Added to Dataframe At The End.
        #ImageAID,ImageAWidth,ImageAHeight,class_0 1 rle_encoded_cell_1_mask class_14 1 rle_encoded_cell_1_mask 0 1 rle encoded_cell_2_mask
        return_dict[ID] = (orig_shape, prediction_str)
        with open(ROOT+'test_p2/'+ID+'.txt', 'w') as f:
            f.write(f'{ID},{orig_shape[0]},{orig_shape[1]},{prediction_str}')
            f.flush()
            os.fsync(f.fileno())
        #if ID == 'cd65456d-cdf6-4f99-805d-bfb85c881b24':
        #    return (orig_shape, default_rle(orig_shape))


def process_image2(ids):
    for ind, ID in enumerate(tqdm(ids)):
        ids_processed = [x.split('.')[0] for x in os.listdir(ROOT+'test_p2/')]
        if ID in ids_processed:
            continue
        print(f'Processing {ID}.')
        img, orig_shape = read_image(ID)
        mask = np.load(ROOT+'test_masks/'+ID+'.npy')
        orig_mask = scale(mask, orig_shape[0], orig_shape[1])
        prediction_str = ''

        for cell in range(1, mask.max()+1):
            mask_bin = np.where(mask == cell, 1, 0)
            img = np.uint8(img*mask_bin[:, :, None])
            #Calculate RLEs for all cells ordered by their ID in mask.
            rle = encode_binary_mask(orig_mask, cell)

            _preds = learn.predict(img)
            class_idxs = np.where(_preds[2]>CONF_THRESH)[0]
            #class_idxs = np.argpartition(_preds[2], -4)[-4:].numpy()

            for class_idx in class_idxs:
                cell_pred = _preds[2][class_idx].item()
                clsid = LEARN_LBL_NAMES[class_idx]
                prediction_str+=f'{int(clsid)} {cell_pred} {rle} '

            #Save Predictions to Be Added to Dataframe At The End.
            #ImageAID,ImageAWidth,ImageAHeight,class_0 1 rle_encoded_cell_1_mask class_14 1 rle_encoded_cell_1_mask 0 1 rle encoded_cell_2_mask
            with open(ROOT+'test_p2/'+ID+'.txt', 'w') as f:
                f.write(f'{ID},{orig_shape[0]},{orig_shape[1]},{prediction_str}')
                f.flush()
                os.fsync(f.fileno())


if __name__ == '__main__':
    print(time.ctime())
    model = cellmodels.Cellpose(gpu=True, model_type='cyto') #) #, net_avg=False, torch=True, device=split
    X_test = [name.rstrip('.png') for name in (os.listdir(ROOT+'public/')) if '.png' in name and name[0] == '1' and name[1] == sys.argv[1]]
    #X_test = ['09852156-bbb8-11e8-b2ba-ac1f6b6435d0'] #'5f79a114-bb99-11e8-b2b9-ac1f6b6435d0', 'fdd126fa-8e87-4ffe-b2b7-3ee48156d887', 'cd65456d-cdf6-4f99-805d-bfb85c881b24', 'cda406d5-178a-4011-b4cc-389ded4dcf27']
    prepare_data(X_test,model)
    #process_image2(X_test)


    """
    X_test = pd.read_csv(ROOT+'../sample_submission.csv').iloc[(split-1)*100:split*100]['ID'].values
    X = np.array_split(X_test, num_processes)
    print(f'Split length: {len(X[0])}.')

    manager = Manager()
    return_dict = manager.dict()
    df = []
    num_processes = 1
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=process_image, args=(X[rank],model,learn,return_dict))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    for k,v in return_dict.items():
        df.loc[df.ID==k,'PredictionString']=v
        df.append([k, v[0][0], v[0][1], v[1]])

    df = pd.DataFrame.from_records(df, columns=['ID', 'ImageWidth', 'ImageHight', 'PredictionString'])
    df.to_csv(ROOT+f'dataset/submission.csv', index=False)
    """
    print(time.ctime())
