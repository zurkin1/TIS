import numpy as np
import os
import shutil
from PIL import Image
import cv2
from shutil import copyfile
import pandas as pd
from pandarallel import pandarallel


pandarallel.initialize()
folder = 'public/publichpa'
ext='.jpg'

def load_image(path):
    R = np.array(Image.open(path+'_red'+ext))[:,:,0]
    G = np.array(Image.open(path+'_green'+ext))[:,:,1]
    B = np.array(Image.open(path+'_blue'+ext))[:,:,2]
    Y = np.array(Image.open(path+'_yellow'+ext))[:,:,0]
    image = np.stack((R/2 + Y/2, G/2 + Y/2, B),-1)
    if image.max()>257:
        image = image/256
    image = Image.fromarray(np.uint8(image))
    image = image.resize((512, 512))
    #image = np.divide(image, 255)
    return image


df = [name.rstrip('green'+ext).rstrip('_') for name in os.listdir(f'./{folder}/') if '_green.' in name]
df = pd.DataFrame(df, columns=['ID'])
#df = pd.read_csv('sample_submission.csv') # train.csv')
#df.columns = ['Id', 'Target']
print(df.head())
#for ind, row in df.iterrows():
#    if ind%100 == 0:
#        print('\r', ind, end="")
#    img = load_image('train/'+row['Id'])
#    cv2.imwrite(f'train_p/{row[0]}.png', img)

def func(row):
    global folder
    if not row[0]+ext in os.listdir(f'./{folder}_p/'):
        img = load_image(f'./{folder}/'+row[0])
        img.save(f'./{folder}_p/{row[0]}{ext}')

df.parallel_apply(func, axis=1)
