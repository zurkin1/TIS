import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pandarallel import pandarallel


pandarallel.initialize()
cell_mask_dir = '/home/zel/dani/data/hpa/mask/hpa_cell_mask'
ROOT = '/home/zel/dani/data/hpa/train/'


def get_cropped_cell(img, msk):
    bmask = msk.astype(int)[...,None]
    masked_img = img * bmask
    true_points = np.argwhere(bmask)
    top_left = true_points.min(axis=0)
    bottom_right = true_points.max(axis=0)
    cropped_arr = masked_img[top_left[0]:bottom_right[0]+1,top_left[1]:bottom_right[1]+1]
    return np.uint8(cropped_arr)


def get_stats(cropped_cell):
    x = (cropped_cell/255.0).reshape(-1,3).mean(0)
    x2 = ((cropped_cell/255.0)**2).reshape(-1,3).mean(0)
    return x, x2


def read_img(image_id, color, image_size=None):
    filename = f'{ROOT}/train/{image_id}_{color}.png'
    assert os.path.exists(filename), f'not found {filename}'
    img= np.array(Image.open(filename))
    #img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if image_size is not None:
        img = cv2.resize(img, (image_size, image_size))
    if img.max() > 255:
        img_max = img.max()
        img = (img/255).astype('uint8')
    return img.astype('uint8')


df = pd.read_csv('train.csv')
labels = [str(i) for i in range(19)]
for x in labels: df[x] = df['Label'].apply(lambda r: int(x in r.split('|')))

dfs_0 = df[df['Label'] == '0'].sample(n=300, random_state=42).reset_index(drop=True)
dfs_1 = df[df['Label'] == '1'].sample(n=221, random_state=42).reset_index(drop=True)
dfs_2 = df[df['Label'] == '2'].sample(n=500, random_state=42).reset_index(drop=True)
dfs_3 = df[df['Label'] == '3'].sample(n=500, random_state=42).reset_index(drop=True)
dfs_4 = df[df['Label'] == '4'].sample(n=500, random_state=42).reset_index(drop=True)
dfs_5 = df[df['Label'] == '5'].sample(n=500, random_state=42).reset_index(drop=True)
dfs_6 = df[df['Label'] == '6'].sample(n=308, random_state=42).reset_index(drop=True)
dfs_7 = df[df['Label'] == '7'].sample(n=500, random_state=42).reset_index(drop=True)
dfs_8 = df[df['Label'] == '8'].sample(n=500, random_state=42).reset_index(drop=True)
dfs_9 = df[df['Label'] == '9'].sample(n=244, random_state=42).reset_index(drop=True)
dfs_10 = df[df['Label'] == '10'].sample(n=310, random_state=42).reset_index(drop=True)
#dfs_11 = df[df['11'] == 1].reset_index(drop=True)
dfs_11 = df[df['Label'] == '11'].sample(n=1, random_state=42).reset_index(drop=True)
dfs_12 = df[df['Label'] == '12'].sample(n=500, random_state=42).reset_index(drop=True)
dfs_13 = df[df['Label'] == '13'].sample(n=400, random_state=42).reset_index(drop=True)
dfs_14 = df[df['Label'] == '14'].sample(n=500, random_state=42).reset_index(drop=True)
dfs_15 = df[df['Label'] == '15'].sample(n=82, random_state=42).reset_index(drop=True)
dfs_16 = df[df['Label'] == '16'].sample(n=350, random_state=42).reset_index(drop=True)
dfs_17 = df[df['Label'] == '17'].sample(n=228, random_state=42).reset_index(drop=True)
dfs_18 = df[df['Label'] == '18'].sample(n=34, random_state=42).reset_index(drop=True)
dfs_ = [dfs_0, dfs_1, dfs_2, dfs_3, dfs_4, dfs_5, dfs_6, dfs_7, dfs_8, dfs_9, dfs_10,
        dfs_11, dfs_12, dfs_13, dfs_14, dfs_15, dfs_16, dfs_17, dfs_18]
dfs = pd.concat(dfs_, ignore_index=True)
dfs.drop_duplicates(inplace=True, ignore_index=True)
print(len(dfs))
print(dfs.Label.value_counts())
dfs = dfs.sample(frac=1, random_state=42).reset_index(drop=True)
dfs['folder'] = 'train'
dfs.loc[dfs.index[0:1000],'folder'] = 'test'

unique_counts = {}
for lbl in labels:
    unique_counts[lbl] = len(dfs[dfs.Label == lbl])

full_counts = {}
for lbl in labels:
    count = 0
    for row_label in dfs['Label']:
        if lbl in row_label.split('|'): count += 1
    full_counts[lbl] = count
    
counts = list(zip(full_counts.keys(), full_counts.values(), unique_counts.values()))
counts = np.array(sorted(counts, key=lambda x:-x[1]))
counts = pd.DataFrame(counts, columns=['label', 'full_count', 'unique_count'])
print(counts.set_index('label').T)

x_tot,x2_tot = [],[]
lbls = []
num_files = len(dfs)
all_cells = []


def process_image(row):
    #for idx in tqdm(range(num_files)):
    image_id = row[0] #dfs.iloc[idx].ID
    label = row[1]
    folder = row[21]
    if not os.path.exists(ROOT+f'/merged/{folder}/{label}'):
                os.mkdir(ROOT+f'/merged/{folder}/{label}')
    cell_mask = np.load(f'{cell_mask_dir}/{image_id}.npz')['arr_0']
    red = read_img(image_id, "red", None)
    green = read_img(image_id, "green", None)
    blue = read_img(image_id, "blue", None)
    #yellow = read_img(image_id, "yellow", train_or_test, image_size)
    #Move from [width, height, channels] to [channels, height, width].
    #stacked_image = np.transpose(np.array([blue, green, red]), (1,2,0))
    stacked_image = np.uint8(np.stack((red, green, blue),-1))

    for cell in range(1, np.max(cell_mask) + 1):
        bmask = cell_mask == cell
        cropped_cell = get_cropped_cell(stacked_image, bmask)
        fname = ROOT+f'/merged/{folder}/{label}/{image_id}_{cell}.png'
        #im = cv2.imencode('.jpg', cropped_cell)[1]
        #img_out.writestr(fname, im)
        #x, x2 = get_stats(cropped_cell)
        #x_tot.append(x)
        #x2_tot.append(x2)
        #all_cells.append({
        #    'image_id': image_id,
        #    'r_mean': x[0],
        #    'g_mean': x[1],
        #    'b_mean': x[2],
        #    'cell_id': cell,
        #    'image_labels': labels,
        #    'size1': cropped_cell.shape[0],
        #    'size2': cropped_cell.shape[1],
        #})
        cropped_cell = Image.fromarray(cropped_cell)
        cropped_cell.save(fname)
    print('done', flush=True)

#image stats
#img_avr =  np.array(x_tot).mean(0)
#img_std =  np.sqrt(np.array(x2_tot).mean(0) - img_avr**2)
#cell_df = pd.DataFrame(all_cells)
#cell_df.to_csv('cell_df.csv', index=False)
#print('mean:',img_avr, ', std:', img_std)
#print(cell_df.head())
#print(cell_df.g_mean.hist(bins=100))
#print(cell_df.r_mean.hist(bins=100))
#print(cell_df.b_mean.hist(bins=100))
dfs.parallel_apply(process_image, axis=1)
dfs.to_csv('train_imgs.csv')