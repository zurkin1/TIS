import numpy as np
import os
import shutil
from PIL import Image
import cv2
from shutil import copyfile
import pandas as pd


def load_image(path, shape=(512,512)):
    R = np.array(Image.open(path+'_red.png'))
    G = np.array(Image.open(path+'_green.png'))
    B = np.array(Image.open(path+'_blue.png'))
    Y = np.array(Image.open(path+'_yellow.png'))

    image = np.stack((
        R/2 + Y/2,
        G/2 + Y/2,
        B),-1)

    image = cv2.resize(image, (shape[0], shape[1]))
    #image = np.divide(image, 255)
    return image


df = pd.read_csv('hpa/train.csv')
df.columns = ['Id', 'Target']
print(df.head())

for ind, row in df.iterrows():
    print(row)
    img = load_image('hpa/'+row['Id'])
    cv2.imwrite(f'hpa_p/{row[0]}.png', img)
