# Extract the best image based on dicom window width and center from each of the structural multi_parametric MRI scans, save that image into PNG and use a pretrained [timm](https://github.com/rwightman/pytorch-image-models/tree/master/timm) model `ig_resnext101_32x16d`.  Inference involves extracting the best image (again based on dicom window width and center) and averaging the prediction across all structural multi_parametric scans.
# This notebook uses various functions from the [fmi](https://github.com/asvcode/fmi) library which is a package that adds additional functionality to [fastai's](https://www.fast.ai/) medical imaging module. 
# To learn more about medical imaging view my [blog](https://asvcode.github.io/MedicalImaging/)
#!git clone https://github.com/asvcode/fmi.git
# Since internet is off in this case you can use the [fmipackage](https://www.kaggle.com/avirdee/fmipackage) dataset
#!cp -r ./fmi/* ./
from fmi.explore import *
from fmi.preprocessing import *
from fmi.pipeline import *
#!pip install timm #'../input/timm034/timm-0.3.4-py3-none-any.whl' -qq
from fastai.vision.all import *
from fastai.medical.imaging import *
import pydicom
from torchvision.utils import save_image
from glob import glob
from skimage import exposure
from pydicom.pixel_data_handlers.util import apply_voi_lut
from timm import create_model
from fastai.vision.learner import _update_first_layer
matplotlib.rcParams['image.cmap'] = 'gist_ncar'


#system_info()

source = '../input/rsna-miccai-brain-tumor-radiogenomic-classification'
train_files = get_dicom_files(f'{source}/train/00000')
train_path = f'{source}/train'
labels = pd.read_csv(f'{source}/train_labels.csv', header=0, names=['id','value'], dtype=object)
#print(os.listdir(source))

"""
# [fmi](https://github.com/asvcode/fmi) has a number of handy features that breakdown the metadata into useful chunks. Firstly you may want to quickly see what image information is contained in the metadata. You can access this using get_image_info
get_image_info(train_files[7])
# Check to see if there is any personally identifiable data in the metadata
get_pii(train_files[7])
# ## Explore Data
# Each dedicated folder is identified by a five-digit number which is further broken down into 4 subfolders:
# 
# - Fluid Attenuated Inversion Recovery (FLAIR)
# - T1-weighted pre-contrast (T1w)
# - T1-weighted post-contrast (T1Gd)
# - T2-weighted (T2)
# 
# Using folder `00000` as an example
source_00000 = f'{source}/train/00000/'
print(os.listdir(source_00000))
# You could view the images within the `T2w` folder like this:
sort_items = get_dicom_files(source_00000, folders='T2w')

imgs = []
for filename in sort_items:
    file = filename.dcmread().pixel_array
    img = TensorDicom(file)
    imgs.append(img)
show_images(imgs, nrows=12)

# Although this displays the images within the folder they are not displayed in sequence. [fmi](https://github.com/asvcode/fmi) provides a handy function that easily sorts the images by `instance number`
instance_show(sort_items, nrows=20)

# MRI scans are made up of numerous 2D slices taken from a number of angles(or planes). The 3 planes are axial, sagittal and cornonal and the image below show what angels the planes represent. [Image credit](https://www.ipfradiologyrounds.com/hrct-primer/image-reconstruction/)
# `show_aspects` is a handy function that easily lets you view the various planes.
# **View all images**
show_aspects(source_00000, show=True, save=False, figsize=(20,10))

# **T2-weighted (T2)**
T2w_00000 = f'{source_00000}/T2w'
show_aspects(T2w_00000, show=True, save=False, figsize=(17,12))

# You can also easily extract the metadata from the dicom images using `from_dicoms`
flair_00000_files = get_dicom_files(FLAIR_00000)

dicom_dataframe = pd.DataFrame.from_dicoms(flair_00000_files, window=dicom_windows.brain, px_summ=True)
dicom_dataframe[:10]
"""

# `get_dicom_image` is another handy feature that lets you view images within a dataframe and sort them by various items such as `img_pct_window`, `img_mean` or `img_std`.
# We will use this later in extracting the best images from each of the folders.
def get_dicom_image(df, key, nrows=1, source=None, folder_val=None, instance_val=None, figsize=(7,7)):
    "Helper to view images by key"
    imgs=[]
    title=[]
    for i in df.index:
        file_path = Path(f"{df.iloc[i]['fname']}")
        dcc = file_path.dcmread().pixel_array
        imgs.append(dcc)
        pct = df.iloc[i][key]
        title.append(pct)
    return show_images(imgs, titles=title, nrows=nrows)

"""
# **sort by `img_pct_window`**
pct = dicom_dataframe[['PatientID', 'InstanceNumber', 'img_pct_window', 'img_mean', 'img_std', 'fname']].sort_values(by=['img_pct_window'], ascending=False).reset_index(drop=True)
get_dicom_image(pct[:10], 'img_pct_window', source=source, nrows=2, figsize=(20,20))

# **sort by `img_mean`**
pct = dicom_dataframe[['PatientID', 'InstanceNumber', 'img_pct_window', 'img_mean', 'img_std', 'fname']].sort_values(by=['img_mean'], ascending=False).reset_index(drop=True)
get_dicom_image(pct[:10], 'img_mean', source=source, nrows=2, figsize=(20,20))

# **sort by `img_std`**
pct = dicom_dataframe[['PatientID', 'InstanceNumber', 'img_pct_window', 'img_mean', 'img_std', 'fname']].sort_values(by=['img_std'], ascending=False).reset_index(drop=True)
get_dicom_image(pct[:10], 'img_std', source=source, nrows=2, figsize=(20,20))

# ### Extract best image from each of the structural multi-parametric MRI (mpMRI) scans
# In order to iterate quickly and get some baseline results the aim is to get the best images from each folder.  In order to do this:
#  - we will extract the dicom metadata from each sub folder
#  - sort the metadata based on dicom window width and center
#  - Create a mask based on the dicom window width and center and get rid of unwanted pixels
#  - Save the image that best matches our criteria by converting it into PNG format
#  - Create a dataframe with columns that contain the `path` to that image (we will need this when constructing the `DataBlock`)
# 
# Note that the `cmap` color used above is purely for aesthetics reasons and actually does not have any impact from a training perspective.  However moving forward we change it to a more appropriate color map
matplotlib.rcParams['image.cmap'] = 'bone'

# `process_dicom` is a handy `fmi` function that crops a dicom image (dependant on dicom window, sigma and thresh values) and saves it in .png format. You have to specify the outpath (where the .png files will be stored), the window width and center, sigma and thresh values.  By default displaying the differences in the images and sanity check is set to `False`
outpath = './test1.png'
window = dicom_windows.brain_soft
sigma = 0.1
thresh = 0.7
remove_max = False

# Have a test run with the first 5 independant cases and using the `T1wCE` folder, we create a dicom dataframe using a `brain_soft` dicom window for each case in the `T1wCE` folder, sort it by `img_pct_window` (img_pct_window is a handy `fastai` function that displays the percentage of pixels within a specified dicom window in an image), and then use `process_dicom` to save the image.
# This is how I created the datasets for each of the structural multi_parametric scans.  To generate the correct paths to save the images I used the code below, which also solves for duplicate file names across the independant cases
"""

def get_outpath(input_dir, dataset, subject, mri):
    img_id = input_dir.split('-')[-1].split('.')[0]
    outpath = os.path.join(f'./{dataset}/{mri}/{subject}.png')
    
    check = os.path.isfile(outpath)
    if check is not True:
        process_dicom(input_dir, outpath, window=dicom_windows.brain_soft, sigma=0.1, thresh=0.7, remove_max=False, show=False, sanity=True)

    return outpath


# We also have to save the image names and file paths so that we can use it in training later
def dicom_to_png(subject):
    fpath = []
    for mri in ['FLAIR', 'T1w', 'T1wCE', 'T2w']:
        path = f'{train_path}/{subject}/{mri}/'
        files = get_dicom_files(path)
        dicom_dataframe = pd.DataFrame.from_dicoms(files, window=dicom_windows.brain_soft, px_summ=True)
        pct = dicom_dataframe[['PatientID', 'InstanceNumber', 'img_pct_window', 'img_mean', 'img_std', 'fname']].sort_values(by=['img_pct_window'], ascending=True).reset_index(drop=True)
        ff = pct['fname'][0]
        outfile = get_outpath(ff, 'train', subject, mri)
        fpath.append(outfile)
    return fpath


labels['t1w_path']= 'path'
labels['t1wce_path'] = 'path'
labels['t2w_path'] = 'path'
labels['flare_path'] = 'path'

#if not os.path.exists('./train/T1wCE'):
for mri in ['FLAIR' ,'T1w', 'T1wCE', 'T2w']:
    if not os.path.exists(f'./train/{mri}'):
        os.makedirs(f'./train/{mri}')
labels['t1wce_path'] = labels.id.apply(lambda x: dicom_to_png(x))
