import numpy as np
import os
import shutil
from PIL import Image
import cv2
from shutil import copyfile
import pandas as pd
from pandarallel import pandarallel


pandarallel.initialize()

def load_image(path):
    R = np.array(Image.open(path+'_red.png'))
    G = np.array(Image.open(path+'_green.png'))
    B = np.array(Image.open(path+'_blue.png'))
    Y = np.array(Image.open(path+'_yellow.png'))

    image = np.stack((R/2 + Y/2, G/2 + Y/2, B),-1)
    image = Image.fromarray(np.uint8(image))
    image = image.resize((512, 512))
    #image = np.divide(image, 255)
    return image


df = pd.read_csv('sample_submission.csv') # train.csv')
df.columns = ['Id', 'Target']
print(df.head())
#for ind, row in df.iterrows():
#    if ind%100 == 0:
#        print('\r', ind, end="")
#    img = load_image('train/'+row['Id'])
#    cv2.imwrite(f'train_p/{row[0]}.png', img)

def func(row):
    img = load_image('./train/'+row[0])
    img.save(f'./train_p/{row[0]}.png')

df.parallel_apply(func, axis=1)