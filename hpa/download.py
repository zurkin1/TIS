import io
import os
import requests
import gzip
import imageio
import pandas as pd
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm
from PIL import Image


def processimg(url):
    img_name = url.split('/')[-1]
    r = requests.get(url, allow_redirects=True)
    img = Image.open(io.BytesIO(r.content))
    #tf = gzip.open(f).read()
    #img = imageio.imread(f, 'jpg')
    img.save('/home/zel/dani/data/hpa/public/images/'+img_name)

    print(f'Downloaded {img_name}')    


if __name__ == '__main__':
    imgList = pd.read_csv('kaggle_2021.tsv')
    # Remove all images overlapping with Training set
    imgList = imgList[imgList.in_trainset == False]

    # Remove all images with only labels that are not in this competition
    imgList = imgList[~imgList.Label_idx.isna()]

    #celllines = ['A-431', 'A549', 'EFO-21', 'HAP1', 'HEK 293', 'HUVEC TERT2', 'HaCaT', 'HeLa', 'PC-3', 'RH-30', 'RPTEC TERT1', 'SH-SY5Y', 'SK-MEL-30', 'SiHa', 'U-2 OS', 'U-251 MG', 'hTCEpi']
    #public_hpa_df_17 = public_hpa_df[public_hpa_df.Cellline.isin(celllines)]
    colors = ['blue', 'red', 'green', 'yellow']
    url_key = []
    for i in imgList['Image']: #Default download all data, for kernel example, I only download 10 image 
        for color in colors:
            img_url = i + "_" + color + ".jpg"
            url_key.append(img_url)
    #processimg(url_key[0])
    pool = ThreadPool(processes=100)
    with tqdm(total=len(url_key)) as bar:
        for _ in pool.imap_unordered(processimg, url_key):
              bar.update(1)
