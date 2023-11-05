
# from importlib.machinery import SourceFileLoader
# utils = SourceFileLoader('stain_utils.py','/content/Stain_Normalization/stain_utils.py' ).load_module()
# stainNorm_Macenko = SourceFileLoader('stainNorm_Macenko.py','/content/Stain_Normalization/stainNorm_Macenko.py' ).load_module()

# Commented out IPython magic to ensure Python compatibility.
from __future__ import division

import stain_utils as utils
#import stainNorm_Reinhard
import stainNorm_Macenko
#import stainNorm_Vahadane
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import os
from PIL import Image

dir = '//home/dsi/zurkin/data26/part2/'


def example():
  i1=utils.read_image('/home/dsi/zurkin/data7/250/TCGA-AO-A1KR-01Z-00-DX1.BFB2E69B-E23C-4542-9CBF-EDD040B985AC.444-12.jpg')
  i2=utils.read_image('/home/dsi/zurkin/data7/250/TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-D8FB-45CF-B4BD-C6B76294C291.0-1.jpg')
  i3=utils.read_image('/home/dsi/zurkin/data7/250/TCGA-3C-AALJ-01Z-00-DX1.777C0957-255A-42F0-9EEB-A3606BCF0C96.658-6.jpg')

  stack=utils.build_stack((i1,i2, i3))

  utils.patch_grid(stack,width=3,save_name='./original.jpg')

  n=stainNorm_Macenko.Normalizer()
  n.fit(i1)
  normalized=utils.build_stack((i1,n.transform(i2), n.transform(i3)))

  utils.patch_grid(normalized,width=3,save_name='./Macenko.jpg')

def load_images():
  pic_list = []
  for i in range(len(os.listdir(dir))):
    i2=utils.read_image(dir + os.listdir(dir)[i])
    print(i)
    pic_list.append(i2)
  return pic_list

def transform(pic_list):
  #i1 is the referens image
  i1 = '/home/dsi/zurkin/data7/all/TCGA-WT-AB41-01Z-00-DX1.75BDFDF2-CD87-46D1-B32C-725741CB02BE.964-22.jpg'
  #i1 = '/home/dsi/zurkin/data_new_one/test/TCGA-OL-A66I-01Z-00-DX1.8CE9DCAB-98D3-4163-94AC-1557D86C1E25.745-19.jpg'
  i1=utils.read_image(i1)
  n=stainNorm_Macenko.Normalizer()
  n.fit(i1)
  result = list(map(lambda x: n.transform(x), pic_list))
  result =  np.array(result)
  return result

def save_to_folder(result):
  for i in range(len(result)):
    im = Image.fromarray(result[i])
    im.save('/home/dsi/zurkin/data27/all/'+ os.listdir(dir)[i])

print("louding images:")
pic_list = load_images()
print("transforming images...")
result = transform(pic_list)
print("saveing images...")
save_to_folder(result)

'''

import numpy as np
from PIL import Image
from __future__ import print_function
import cv2
import sys
import numpy

numpy.set_printoptions(threshold=sys.maxsize)
image_arr = cv2.imread(("/content/drive/My Drive/1m.tif"), 1)
print(image_arr)
indices = np.where(image_arr == [255])
coordinates = list(zip(indices[0], indices[1]))
print (coordinates)

zeros = np.zeros((100, 100, 3), dtype=np.uint8)
zeros[:5,:5,1] = 255

indices = np.where(zeros == [255])

coordinates = list(zip(indices[0], indices[1]))
print (coordinates)

import numpy as np
from sklearn import metrics


y = np.array([2,3,1,0])
pred = np.array([[2.3],[3.5],[0.8],[0.2]])

MAS = metrics.mean_absolute_error(pred,y)

R2 = metrics.r2_score(pred,y)

MAS

R

import pandas as pd

data = [['New York Yankees', 'Acevedo Juan', 900000, 'Pitcher'], 
        ['New York Yankees', 'Anderson Jason', 300000, 'Pitcher'], 
        ['New York Yankees', 'Clemens Roger', 10100000, 'Pitcher'], 
        ['New York Yankees', 'Contreras Jose', 5500000, 'Pitcher']]

df = pd.DataFrame.from_records(data)

df

path = '/content/drive/My Drive/Draft/coordinates.csv'

location = pd.read_csv(path,names=['image','location'])

location

last_pic = ''
lists = []
image_list = []
for index, row in location.iterrows():
    pic = row['image']
    if last_pic != pic:
      lists.append(image_list)
      last_pic = pic
      image_list = [pic, row['location']]
    else:
      image_list.append(row['location'])
lists.append(image_list)

lists= lists[1:]

print(lists)

df = pd.DataFrame.from_records(lists)

df

df.to_csv (r'/content/drive/My Drive/Draft/export_dataframe.csv', index = False, header=True)
'''