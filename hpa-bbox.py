import os
from pathlib import Path
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import argparse
from PIL import Image
from torchvision import transforms
import os
import fastai
import hpacellseg.cellsegmentator as cellsegmentator
from hpacellseg.utils import label_cell
import pandas as pd
import matplotlib.pyplot as plt
import ast
from tqdm import tqdm
import torch
from pycocotools import _mask as coco_mask
import zlib
import base64
from fastai.vision.all import *
from fastai.metrics import error_rate
from fastai.distributed import *

########################### Create bbox.
torch.cuda.set_device(1)
path = '/home/dsi/zurkin/data/'
protein_stats = ([0.08069, 0.05258, 0.05487], [0.13704, 0.10145, 0.15313])
src = ImageDataLoaders.from_folder(path+'hpa/', valid_pct=0.05, # bs=64,
                                batch_tfms=aug_transforms(do_flip=True, flip_vert=True, max_rotate=180.0, max_lighting=0.4, p_affine=0.7, p_lighting=0.7, xtra_tfms=Normalize.from_stats(*protein_stats)))
learn = cnn_learner(
    src,
    alexnet, #resnet18,
    #cut=-2,
    #splitter=_resnet_split,
    #loss_func=F.binary_cross_entropy_with_logits,
    path=path+'hpa/',
    #metrics=[f1score_multi]
)
learn.load('stage-2-rn18')


def binary_mask_to_ascii(mask, mask_val=1):
    """Converts a binary mask into OID challenge encoding ascii text."""
    mask = np.where(mask==mask_val, 1, 0).astype(np.bool)

    # check input mask --
    if mask.dtype != np.bool:
        raise ValueError(f"encode_binary_mask expects a binary mask, received dtype == {mask.dtype}")

    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError(f"encode_binary_mask expects a 2d mask, received shape == {mask.shape}")

    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

    # compress and base64 encoding --
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str.decode()


def rle_encoding(img, mask_val=1):
    """
    Turns our masks into RLE encoding to easily store them
    and feed them into models later on
    https://en.wikipedia.org/wiki/Run-length_encoding

    Args:
        img (np.array): Segmentation array
        mask_val (int): Which value to use to create the RLE

    Returns:
        RLE string

    """
    dots = np.where(img.T.flatten() == mask_val)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b

    return ' '.join([str(x) for x in run_lengths])


def rle_to_mask(rle_string, height, width):
    """ Convert RLE sttring into a binary mask

    Args:
        rle_string (rle_string): Run length encoding containing
            segmentation mask information
        height (int): Height of the original image the map comes from
        width (int): Width of the original image the map comes from

    Returns:
        Numpy array of the binary segmentation mask for a given cell
    """
    rows,cols = height,width
    rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]
    rle_pairs = np.array(rle_numbers).reshape(-1,2)
    img = np.zeros(rows*cols,dtype=np.uint8)
    for index,length in rle_pairs:
        index -= 1
        img[index:index+length] = 255
    img = img.reshape(cols,rows)
    img = img.T
    return img


def decode_img(img, img_size=(224,224), testing=False):
    """TBD"""
    
    # convert the compressed string to a 3D uint8 tensor
    if not testing:
        # resize the image to the desired size
        img = tf.image.decode_png(img, channels=1)
        return tf.cast(tf.image.resize(img, img_size), tf.uint8)
    else:
        return tf.image.decode_png(img, channels=1)


def preprocess_path_ds(rp, gp, bp, yp, lbl, n_classes=19, img_size=(224,224), combine=True, drop_yellow=True):
    """ TBD """
    
    ri = decode_img(tf.io.read_file(rp), img_size)
    gi = decode_img(tf.io.read_file(gp), img_size)
    bi = decode_img(tf.io.read_file(bp), img_size)
    yi = decode_img(tf.io.read_file(yp), img_size)

    if combine and drop_yellow:
        return tf.stack([ri[..., 0], gi[..., 0], bi[..., 0]], axis=-1), tf.one_hot(lbl, n_classes, dtype=tf.uint8)
    elif combine:
        return tf.stack([ri[..., 0], gi[..., 0], bi[..., 0], yi[..., 0]], axis=-1), tf.one_hot(lbl, n_classes, dtype=tf.uint8)
    elif drop_yellow:
        return ri, gi, bi, tf.one_hot(lbl, n_classes, dtype=tf.uint8)
    else:
        return ri, gi, bi, yi, tf.one_hot(lbl, n_classes, dtype=tf.uint8)


def load_image(img_id, img_dir, testing=False):
    """ Load An Image Using ID and Directory Path - Composes 4 Individual Images """
    if not testing:
        rgby = [
            np.asarray(Image.open(os.path.join(img_dir, img_id+f"_{c}.png")), np.uint8) \
            for c in ["red", "green", "blue", "yellow"]
        ]
        return np.stack(rgby, axis=-1)
    else:
        # This is for cellsegmentator
        return np.stack(
            [np.asarray(Image.open(os.path.join(img_dir, img_id+f"_{c}.png")), \
                        #decode_img(tf.io.read_file(os.path.join(img_dir, img_id+f"_{c}.png")), testing=True), \
                        np.uint8) \
             for c in ["red", "green", "blue", "yellow"]], axis=0
        )


def plot_rgb(arr, figsize=(12,12)):
    """ Plot 3 Channel Microscopy Image """
    plt.figure(figsize=figsize)
    plt.title(f"RGB Composite Image", fontweight="bold")
    plt.imshow(arr)
    plt.axis(False)
    plt.show()


def convert_rgby_to_rgb(arr):
    """ Convert a 4 channel (RGBY) image to a 3 channel RGB image.

    Advice From Competition Host/User: lnhtrang

    For annotation (by experts) and for the model, I guess we agree that individual
    channels with full range px values are better.
    In annotation, we toggled the channels.
    For visualization purpose only, you can try blending the channels.
    For example,
        - red = red + yellow
        - green = green + yellow/2
        - blue=blue.

    Args:
        arr (numpy array): The RGBY, 4 channel numpy array for a given image

    Returns:
        RGB Image
    """

    rgb_arr = np.zeros_like(arr[..., :-1])
    rgb_arr[..., 0] = arr[..., 0]
    rgb_arr[..., 1] = arr[..., 1]+arr[..., 3]/2
    rgb_arr[..., 2] = arr[..., 2]

    return rgb_arr


def plot_ex(arr, figsize=(20,6), title=None, plot_merged=True, rgb_only=False):
    """ Plot 4 Channels Side by Side """
    if plot_merged and not rgb_only:
        n_images=5 
    elif plot_merged and rgb_only:
        n_images=4
    elif not plot_merged and rgb_only:
        n_images=4
    else:
        n_images=3
    plt.figure(figsize=figsize)
    if type(title) == str:
        plt.suptitle(title, fontsize=20, fontweight="bold")

    for i, c in enumerate(["Red Channel – Microtubles", "Green Channel – Protein of Interest", "Blue - Nucleus", "Yellow – Endoplasmic Reticulum"]):
        if not rgb_only:
            ch_arr = np.zeros_like(arr[..., :-1])        
        else:
            ch_arr = np.zeros_like(arr)
        if c in ["Red Channel – Microtubles", "Green Channel – Protein of Interest", "Blue - Nucleus"]:
            ch_arr[..., i] = arr[..., i]
        else:
            if rgb_only:
                continue
            ch_arr[..., 0] = arr[..., i]
            ch_arr[..., 1] = arr[..., i]
        plt.subplot(1,n_images,i+1)
        plt.title(f"{c.title()}", fontweight="bold")
        plt.imshow(ch_arr)
        plt.axis(False)
        
    if plot_merged:
        plt.subplot(1,n_images,n_images)
        
        if rgb_only:
            plt.title(f"Merged RGB", fontweight="bold")
            plt.imshow(arr)
        else:
            plt.title(f"Merged RGBY into RGB", fontweight="bold")
            plt.imshow(convert_rgby_to_rgb(arr))
        plt.axis(False)
        
    plt.tight_layout(rect=[0, 0.2, 1, 0.97])
    plt.show()


def create_segmentation_maps(list_of_image_lists, segmentator, batch_size=8):
    """ Function to generate segmentation maps using CellSegmentator tool 
    
    Args:
        list_of_image_lists (list of lists):
            - [[micro-tubules(red)], [endoplasmic-reticulum(yellow)], [nucleus(blue)]]
        batch_size (int): Batch size to use in generating the segmentation masks
        
    Returns:
        List of lists containing RLEs for all the cells in all images
    """
    
    all_mask_rles = {}
    for i in tqdm(range(0, len(list_of_image_lists[0]), batch_size), total=len(list_of_image_lists[0])//batch_size):
        
        # Get batch of images
        sub_images = [img_channel_list[i:i+batch_size] for img_channel_list in list_of_image_lists] # 0.000001 seconds

        # Do segmentation
        cell_segmentations = segmentator.pred_cells(sub_images)
        nuc_segmentations = segmentator.pred_nuclei(sub_images[2])

        # post-processing
        for j, path in enumerate(sub_images[0]):
            img_id = path.replace("_red.png", "").rsplit("/", 1)[1]
            nuc_mask, cell_mask = label_cell(nuc_segmentations[j], cell_segmentations[j])
            new_name = os.path.basename(path).replace('red','mask')
            all_mask_rles[img_id] = [rle_encoding(cell_mask, mask_val=k) for k in range(1, np.max(cell_mask)+1)]
    return all_mask_rles


def get_img_list(img_dir, return_ids=False, sub_n=None):
    """ Get image list in the format expected by the CellSegmentator tool """
    if sub_n is None:
        sub_n=len(glob(img_dir + '/' + f'*_red.png'))
    if return_ids:
        images = [sorted(glob(img_dir + '/' + f'*_{c}.png'))[:sub_n] for c in ["red", "yellow", "blue"]]
        return [x.replace("_red.png", "").rsplit("/", 1)[1] for x in images[0]], images
    else:
        return [sorted(glob(img_dir + '/' + f'*_{c}.png'))[:sub_n] for c in ["red", "yellow", "blue"]]
    
    
def get_contour_bbox_from_rle(rle, width, height, return_mask=True,):
    """ Get bbox of contour as `xmin ymin xmax ymax`
    
    Args:
        rle (rle_string): Run length encoding containing 
            segmentation mask information
        height (int): Height of the original image the map comes from
        width (int): Width of the original image the map comes from
    
    Returns:
        Numpy array for a cell bounding box coordinates
    """
    mask = rle_to_mask(rle, height, width).copy()
    cnts = grab_contours(
        cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        ))
    x,y,w,h = cv2.boundingRect(cnts[0])
    
    if return_mask:
        return x,y,x+w,y+h, mask
    else:
        return x,y,x+w,y+h


def flatten_list_of_lists(l_o_l, to_string=False):
    if not to_string:
        return [item for sublist in l_o_l for item in sublist]
    else:
        return [str(item) for sublist in l_o_l for item in sublist]


def get_contour_bbox_from_raw(raw_mask):
    """ Get bbox of contour as `xmin ymin xmax ymax`

    Args:
        raw_mask (nparray): Numpy array containing segmentation mask information

    Returns:
        Numpy array for a cell bounding box coordinates
    """
    cnts = grab_contours(
        cv2.findContours(
            raw_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        ))
    xywhs = [cv2.boundingRect(cnt) for cnt in cnts]
    xys = [(xywh[0], xywh[1], xywh[0]+xywh[2], xywh[1]+xywh[3]) for xywh in xywhs]
    return sorted(xys, key=lambda x: (x[1], x[0]))


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


def cut_out_cells(rgby, rles, resize_to=(256,256), square_off=True, return_masks=False, from_raw=True):
    """ Cut out the cells as padded square images 
    
    Args:
        rgby (np.array): 4 Channel image to be cut into tiles
        rles (list of RLE strings): List of run length encoding containing 
            segmentation mask information
        resize_to (tuple of ints, optional): The square dimension to resize the image to
        square_off (bool, optional): Whether to pad the image to a square or not
        
    Returns:
        list of square arrays representing squared off cell images
    """
    w,h = rgby.shape[:2]
    contour_bboxes = [get_contour_bbox(rle, w, h, return_mask=return_masks) for rle in rles]
    if return_masks:
        masks = [x[-1] for x in contour_bboxes]
        contour_bboxes = [x[:-1] for x in contour_bboxes]
    
    arrs = [rgby[bbox[1]:bbox[3], bbox[0]:bbox[2], ...] for bbox in contour_bboxes]
    if square_off:
        arrs = [pad_to_square(arr) for arr in arrs]
        
    if resize_to is not None:
        arrs = [
            cv2.resize(pad_to_square(arr).astype(np.float32), 
                       resize_to, 
                       interpolation=cv2.INTER_CUBIC) \
            for arr in arrs
        ]
    if return_masks:
        return arrs, masks
    else:
        return arrs


def grab_contours(cnts):
    # if the length the contours tuple returned by cv2.findContours
    # is '2' then we are using either OpenCV v2.4, v4-beta, or
    # v4-official
    if len(cnts) == 2:
        cnts = cnts[0]

    # if the length of the contours tuple is '3' then we are using
    # either OpenCV v3, v4-pre, or v4-alpha
    elif len(cnts) == 3:
        cnts = cnts[1]

    # otherwise OpenCV has changed their cv2.findContours return
    # signature yet again and I have no idea WTH is going on
    else:
        raise Exception(("Contours tuple must have length 2 or 3, "
            "otherwise OpenCV changed their cv2.findContours return "
            "signature yet again. Refer to OpenCV's documentation "
            "in that case"))

    # return the actual contours array
    return cnts


def preprocess_row(img_id, img_w, img_h, combine=True, drop_yellow=True):
    """ TBD """

    rp = os.path.join(TEST_IMG_DIR, img_id+"_red.png")
    gp = os.path.join(TEST_IMG_DIR, img_id+"_green.png")
    bp = os.path.join(TEST_IMG_DIR, img_id+"_blue.png")
    yp = os.path.join(TEST_IMG_DIR, img_id+"_yellow.png")
    
    ri = decode_img(tf.io.read_file(rp), (img_w, img_h), testing=True)
    gi = decode_img(tf.io.read_file(gp), (img_w, img_h), testing=True)
    bi = decode_img(tf.io.read_file(bp), (img_w, img_h), testing=True)

    if not drop_yellow:
        yi = decode_img(tf.io.read_file(yp), (img_w, img_h), testing=True)

    if combine and drop_yellow:
        return tf.stack([ri[..., 0], gi[..., 0], bi[..., 0]], axis=-1)
    elif combine:
        return tf.stack([ri[..., 0], gi[..., 0], bi[..., 0], yi[..., 0]], axis=-1)
    elif drop_yellow:
        return ri, gi, bi
    else:
        return ri, gi, bi, yi


def plot_predictions(img, masks, preds, confs=None, fill_alpha=0.3, lbl_as_str=True):
    # Initialize
    FONT = cv2.FONT_HERSHEY_SIMPLEX; FONT_SCALE = 0.7; FONT_THICKNESS = 2; FONT_LINE_TYPE = cv2.LINE_AA;
    COLORS = [[round(y*255) for y in x] for x in sns.color_palette("Spectral", len(LBL_NAMES))]
    to_plot = img.copy()
    cntr_img = img.copy()
    if confs==None:
        confs = [None,]*len(masks)

    cnts = grab_contours(
        cv2.findContours(
            masks,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        ))
    cnts = sorted(cnts, key=lambda x: (cv2.boundingRect(x)[1], cv2.boundingRect(x)[0]))

    for c, pred, conf in zip(cnts, preds, confs):
        # We can only display one color so we pick the first
        color = COLORS[pred[0]]
        if not lbl_as_str:
            classes = "CLS=["+",".join([str(p) for p in pred])+"]"
        else:
            classes = ", ".join([INT_2_STR[p] for p in pred])
        M = cv2.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        text_width, text_height = cv2.getTextSize(classes, FONT, FONT_SCALE, FONT_THICKNESS)[0]

        # Border and fill
        cv2.drawContours(to_plot, [c], contourIdx=-1, color=[max(0, x-40) for x in color], thickness=10)
        cv2.drawContours(cntr_img, [c], contourIdx=-1, color=(color), thickness=-1)

        # Text
        cv2.putText(to_plot, classes, (cx-text_width//2,cy-text_height//2),
                    FONT, FONT_SCALE, [min(255, x+40) for x in color], FONT_THICKNESS, FONT_LINE_TYPE)

    cv2.addWeighted(cntr_img, fill_alpha, to_plot, 1-fill_alpha, 0, to_plot)
    plt.figure(figsize=(16,16))
    plt.imshow(to_plot)
    plt.axis(False)
    plt.show()


if __name__ == '__main__':
    NUC_MODEL = 'dpn_unet_nuclei_v1.pth'
    CELL_MODEL = 'dpn_unet_cell_3ch_v1.pth'
    IMAGE_SIZES = [1728, 2048, 3072]
    BATCH_SIZE = 8
    CONF_THRESH = 0.25
    RELABEL_UNCERTAIN = True
    TILE_SIZE = (256,256)
    TEST_IMG_DIR = path+'/hpa_test'

    # Make subset dataframes
    sub_df = pd.read_csv(path+'sample_submission.csv')
    sub_df_1728 = sub_df[sub_df.ImageWidth==IMAGE_SIZES[0]]
    sub_df_2048 = sub_df[sub_df.ImageWidth==IMAGE_SIZES[1]]
    sub_df_3072 = sub_df[sub_df.ImageWidth==IMAGE_SIZES[2]]
    submission_ids_1728 = sub_df_1728.ID.to_list()
    submission_ids_2048 = sub_df_2048.ID.to_list()
    submission_ids_3072 = sub_df_3072.ID.to_list()

    predictions = []
    test_df = pd.DataFrame(columns=["ID"], data=submission_ids_1728+submission_ids_2048+submission_ids_3072)
    #segmentator = cellsegmentator.CellSegmentator(path+NUC_MODEL, path+CELL_MODEL, scale_factor=0.25, padding=True, device='cuda')

    for submission_ids in [submission_ids_1728, submission_ids_2048, submission_ids_3072]:
        for i in tqdm(range(0, len(submission_ids), BATCH_SIZE), total=int(np.ceil(len(submission_ids)/BATCH_SIZE))):

            # Step 0: Get batch of images as numpy arrays
            batch_rgby_images = [
                #path+'hpa_test/'+ID+".png" \
                #Image.open(os.path(path+'test/'+ID+".png")) \
                load_image(ID, TEST_IMG_DIR, testing=True) \
                for ID in submission_ids[i:(i+BATCH_SIZE)]
            ]

            # Step 1: Do Prediction On Batch
            #cell_segmentations = segmentator.pred_cells([[rgby_image[j] for rgby_image in batch_rgby_images] for j in [0, 3, 2]])
            #nuc_segmentations = segmentator.pred_nuclei([rgby_image[2] for rgby_image in batch_rgby_images])
            # Perform Cell Labelling on Batch
            #batch_masks = [label_cell(nuc_seg, cell_seg)[1].astype(np.uint8) for nuc_seg, cell_seg in zip(nuc_segmentations, cell_segmentations)]
            #[np.savez_compressed(path+'/hpa_test_masks/'+filename, mask) for (filename, mask) in zip(submission_ids[i:(i+BATCH_SIZE)], batch_masks)]

            # Step 2: Alternatively load masks from disk.
            batch_masks = [np.load(path+'/hpa_test_masks/'+filename+'.npz')['arr_0'] for filename in submission_ids[i:(i+BATCH_SIZE)]]
            
            # Step 3: Reshape the RGBY Images so They Are Channels Last Across the Batch
            batch_rgb_images = [rgby_image.transpose(1,2,0)[..., :-1] for rgby_image in batch_rgby_images]

            # Step 4: Generate Submission RLEs For the Batch
            submission_rles = [[binary_mask_to_ascii(mask, mask_val=cell_id) for cell_id in range(1, mask.max()+1)] for mask in batch_masks]

            # Step 5: Get Bounding Boxes For All Cells in All Images in Batch
            batch_cell_bboxes = [get_contour_bbox_from_raw(mask) for mask in batch_masks]

            # Step 6: Cut Out, Pad to Square, and Resize.
            batch_cell_tiles = [[
                cv2.resize(
                    pad_to_square(
                        rgb_image[bbox[1]:bbox[3], bbox[0]:bbox[2], ...]),
                    TILE_SIZE, interpolation=cv2.INTER_CUBIC) for bbox in bboxes]
                for bboxes, rgb_image in zip(batch_cell_bboxes, batch_rgb_images)
            ]

            # Step 7: Perform Inference
            batch_o_preds = [[learn.predict(tile) for tile in cell_tiles] for cell_tiles in batch_cell_tiles]
            #preds,_ = learn.get_preds(DatasetType.Test)
            #pred_labels = [' '.join(list([str(i) for i in np.nonzero(row>0.2)[0]])) for row in np.array(preds)]

            # Step 8: Post-Process
            batch_confs = [[cell[2][np.where(cell[2]>CONF_THRESH)] for cell in image_preds] for image_preds in batch_o_preds]
            batch_preds = [[np.where(cell[2]>CONF_THRESH)[0] for cell in image_preds] for image_preds in batch_o_preds]
            #if RELABEL_UNCERTAIN:
            #    for j, preds in enumerate(batch_preds):
            #        for k in range(len(preds)):
            #            if preds[k].size==0:
            #                batch_preds[j][k]=np.array([18,])
            #                batch_confs[j][k]=np.array([1-np.max(batch_o_preds[j][k]),])

            # Optional Viz Step
            #print("\n... DEMO IMAGE ...\n")
            #plot_predictions(batch_rgb_images[0], batch_masks[0], batch_preds[0], confs=batch_confs[0], fill_alpha=0.2, lbl_as_str=True)

            # Step 9: Format Predictions To Create Prediction String Easily
            submission_rles = [flatten_list_of_lists([[m,]*len(p) for m, p in zip(masks, preds)]) for masks, preds in zip(submission_rles, batch_preds)]
            batch_preds = [flatten_list_of_lists(preds, to_string=True) for preds in batch_preds]
            batch_confs = [[f"{conf:.4f}" for cell_confs in confs for conf in cell_confs] for confs in batch_confs]

            # Step 10: Save Predictions to Be Added to Dataframe At The End
            # ImageAID,ImageAWidth,ImageAHeight,class_0 1 rle_encoded_cell_1_mask class_14 1 rle_encoded_cell_1_mask 0 1 rle encoded_cell_2_mask
            predictions.extend([" ".join(flatten_list_of_lists(zip(*[preds,confs,masks]))) for preds, confs, masks in zip(batch_preds, batch_confs, submission_rles)])

    test_df["PredictionString"] = predictions
    test_df.to_csv(path+'submission.csv')