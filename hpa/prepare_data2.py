import numpy as np
import os
import shutil
from PIL import Image
import cv2
from shutil import copyfile
import pandas as pd
from pandarallel import pandarallel


pandarallel.initialize()
ROOT = '/home/zel/dani/data/hpa/train/train/'

def load_image(row):
    global df_single
    ID = row[0].split('_')[0]
    if ID not in set(df_single.ID.values):
        return
    name = row[0][:-8]
    for cell_class in ['actin_filaments', 'aggresome', 'centrosome', 'cytosol', 'endoplasmic_reticulum', 'golgi_apparatus', 'intermediate_filaments', 'microtubules', 'mitochondria', 'mitotic_spindle', 'negative', 'nuclear_bodies', 'nuclear_membrane', 'nuclear_speckles', 'nucleoli', 'nucleoli_fibrillar_center', 'nucleoplasm', 'plasma_membrane', 'vesicles']:
    if(os.path.exists(ROOT+f'red/red_{cell_class}_256/data/train_tiles/{cell_class}/{row[0]}')):
        R = np.array(Image.open(ROOT+f'red/red_{cell_class}_256/data/train_tiles/{cell_class}/{row[0]}'))
        G = np.array(Image.open(ROOT+f'gren/green_{cell_class}_256/data/train_tiles/{cell_class}/{row[0]}'))
        B = np.array(Image.open(ROOT+f'blue/blue_{cell_class}_256/data/train_tiles/{cell_class}/{row[0]}'))
        #Y = np.array(Image.open(ROOT+f'yellow/yellow_{cell_class}_256/data/train_tiles/{cell_class}/{name}_yellow.png'))
        #image = np.stack((R/2 + Y/2, G/2 + Y/2, B),-1)
        image = Image.fromarray(np.uint8(np.stack((R, G, B),-1)))
        #image = cv2.resize(image, (shape[0], shape[1]))
        #image = np.divide(image, 255)
        #return image
        if not os.path.exists(ROOT+f'train'):
            os.mkdir(ROOT+f'train')
        #cv2.imwrite(ROOT+f'train/{cell_class}/{row[0][:-8]}.png', image)
        image.save(ROOT+f'train/{cell_class}/{row[0][:-8]}.png')

df = pd.read_csv('train_single_class.csv')
df.parallel_apply(load_image, axis=1)