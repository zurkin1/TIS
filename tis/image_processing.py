import numpy as np
#import openslide
import os
import shutil
from PIL import Image
import cv2
#from openslide import deepzoom
from shutil import copyfile
from heapq import nlargest
import pandas as pd
#from openslide.lowlevel import *
#import operator


def max_min_in_data():
    """checks what is the max and min width and hight in the data"""
    directory = '/home/zel/rimon/Ray/bin/TIS/images_svs/'
    max_img_levels = 0
    min_img_levels = 10
    max_img_dim = 0
    min_img_dim = 9999999999
    count_file = 0
    for filename in os.listdir(directory):
        if count_file % 100 == 0:
            print(count_file)
        count_file += 1
        if filename.endswith(".svs"):
            image1 = openslide.OpenSlide(directory + filename)
            count = image1.level_count                                 #number of levels
            dim = image1.level_dimensions[2][0]                         #(width, height) of the 2nd level
            if count > max_img_levels:
                max_img_levels = count
            if count < min_img_levels:
                min_img_levels = count
            if dim > max_img_dim:
                max_img_dim = dim
            if dim < min_img_dim:
                min_img_dim = dim
        else:
            continue
    print(max_img_levels, min_img_levels, max_img_dim, min_img_dim)



def convert_svs_to_jpg_in_folder(level, dir_in, dir_out):
    """convert all svs files in the folder to jpg in the level dimention entered"""
    num = 0
    for filename in os.listdir(dir_in):
        num = num + 1
        try:
            convert_svs_to_jpg(level, filename, dir_in, dir_out)
            print(num, filename)
        except:
            continue


def convert_svs_to_jpg(level, filename, dir_in, dir_out):
    """convert only one svs file to jpg in the level dimention entered"""

    image1 = openslide.OpenSlide(dir_in + filename)
    dim0 = image1.level_dimensions[level]
    print(dim0)
    RGBA_image = image1.read_region((0, 0), level,dim0)  # convert to RGBA image: top left corner=(0,0),  size= all image dim
    RGB_image = RGBA_image.convert(mode='RGB', matrix=None, dither=None, palette=0, colors=256)
    new_filename = filename[:-3] + 'jpg'
    RGB_image.save(dir_out + new_filename)


def deep_zoom_tiles_in_folder(dir_in, dir_out):
    """takes SVS files and cut to JPG tils, puts all tills from the same svs pic in the same folder, uses the 2nd level = 10X"""
    count = 0

    for filename in os.listdir(dir_in):
        print(count, filename)
        folder_num = dir_out + str(count) +"/"
        #os.mkdir(folder_num)
        deep_zoom_tiles(dir_in, folder_num, filename, dir_out)
        count += 1


def deep_zoom_tiles(dir_in, folder_num, filename, dir_out ):
    """takes only one SVS file and cut to JPG tile, puts all tills in the same new folder, uses the 2nd level = 10X"""

    image = openslide.OpenSlide(dir_in + filename)
    image_deep = deepzoom.DeepZoomGenerator(image, tile_size=512, overlap=0, limit_bounds=True)
    num_tils = (image_deep.level_tiles[image_deep.level_count-1][0]) * (image_deep.level_tiles[image_deep.level_count-1][1])  #the number of all tiles

    num = 0
    for i in range(image_deep.level_tiles[image_deep.level_count-1][0]):          # level_count-2 = 1 mpp = 10X            level_count-1 = 0.5 mpp = 20X
        for j in range(image_deep.level_tiles[image_deep.level_count-1][1]):        # level_tiles[image_deep.level_count-2][0] takes the tiles on the width, level_tiles[image_deep.level_count-2][1] takes the tiles on the hight
            RGB_image = image_deep.get_tile(image_deep.level_count-1, (i, j))
            new_filename = filename[:-4] + str(folder_num) +'.jpg'
            RGB_image.save(dir_out + folder_num + new_filename)
            print(num, "out of:" , num_tils)
            num +=1


def deep_zoom_tiles_one(Coordinates,size, name, dir_in):
    """takes only one SVS file and cut to only one JPG tile"""

    filename  = dir_in + name
    image = openslide.OpenSlide(filename)
    RGBA_image = image.read_region(Coordinates,1,size)
    RGB_image = RGBA_image.convert(mode='RGB', matrix=None, dither=None, palette=0, colors=256)
    return RGB_image


def crop_jpg(height, width, filename, dir_in, dir_out):
    """takes JPG and cut to JPG tiles, height= the height of the tile, width= the width of the tile"""

    n = 0
    im = Image.open(dir_in + filename)
    imgwidth, imgheight = im.size
    for i in range(0, imgheight, height):
            if i + height > imgheight:
                break
            for j in range(0, imgwidth, width):
                if j+width>imgwidth:
                    break
                n += 1
                box = (j, i, j+width, i+height)
                a = im.crop(box)
                a.save("%s%s-%s.jpg" % (dir_out,filename[:-4], n))

def drop_background_tiles_in_folder(threshold_pix, threshold_percent, dir_in, dir_out):
    """drops tiles with more than 'threshold_percent' background in each folder
    threshold_pix= decide what value of pixel considered a white background
    image_size= the width( and length) of the image
    threshold_percent= the percent of the background
    white= the number of white pixels
    black= the number of black pixels"""

    for folder in os.listdir(dir_in):
        if int(folder) > 65:
            for filename in os.listdir(dir_in + folder):
                drop_background(threshold_pix, threshold_percent, filename, folder, dir_in, dir_out)


def drop_background(threshold_pix, threshold_percent, filename, dir_in, dir_out):
    """drops tiles with more than 'threshold_percent' background.
    threshold_pix= decide what value of pixel considered a white background
    image_size= the width( and length) of the image
    threshold_percent= the percent of the background
    white= the number of white pixels
    black= the number of black pixels"""

    img_array = cv2.imread(os.path.join(dir_in, filename), 1) #load a colored image
    #print(img_array)
    #print(img_array.shape)
    '''
    matrix_black = (img_array != 0).sum(2) #Count how many non-zeros are in each pixel
    black = np.count_nonzero(matrix_black == 0)
    matrix_white = (img_array <= threshold_pix).sum(2) #Count how many non-zeros are in each pixel
    white = np.count_nonzero(matrix_white == 0)
    #print('#####################')
    percent = ((black+white)/(1500*1500))*100
    #print("file: %s, Percent: %s, white: %s, black: %s" %(filename[-10:],percent,white/22500,black/22500))
    if percent <= threshold_percent:
        im = Image.open(dir_in  + filename)
        im.save(dir_out +filename)
    '''
def rotate(degree, dir_in, dir_out ):
    for filename in os.listdir(dir_in):
        print(filename)
        image = Image.open(dir_in + filename)
        ima = image.rotate(degree)
        ima.save(dir_out+filename[:-3]+"down"+".jpg")

def make_folders(dir):
    for i in range(175):
        os.mkdir(dir + str(i))


def k_max_images(k, dir_in, dir_out):
    """ saves the k pictures with the highest GB size in the dir_out directory"""

    size_list = []

    for folder in os.listdir(dir_in):
        for file in os.listdir(dir_in + folder + "/"):
            size = os.path.getsize(dir_in + folder + "/" + file)
            size_list.append((file, size))                                     #makes a list of (picture name,picture size) tuples for evrey picture inside a folder

        size_list.sort(key=lambda x: x[1], reverse=True)                        #sort by the "size" argument, from largest to smallest
        os.mkdir(dir_out + folder)                                        # creates new folder in 'directory_out' with the same folder name
        print(folder)
        if len(size_list) > k:
            for i in range(k):
                copyfile(dir_in + folder + '/' + size_list[i][0], dir_out + folder + '/' + size_list[i][0])
                print(size_list[i][1])
        else:
            for i in range(len(size_list)):
                copyfile(dir_in + folder + '/' + size_list[i][0], dir_out + folder + '/' + size_list[i][0])
                print(size_list[i][1])
        size_list = []

def k_max_images_to_csv(dir_in):
    """ saves the k pictures with the highest GB size in the dir_out directory"""
    size_dict = {}

    for filename in os.listdir(dir_in):
        ID = (filename.split(".")[-2]).split("-")[-2]
#        patch = (filename.split(".")[-2]).split("-")[-1]
        size = os.path.getsize(dir_in+filename)
        if ID not in size_dict.keys():
            size_dict[ID] = []
        else:
            size_dict[ID].append((filename, size))
                                           #makes a list of (picture name,picture size) tuples for evrey picture inside a folder
    for key in size_dict.keys():
        size_dict[key].sort(key=lambda x: x[1], reverse=True)                        #sort by the "size" argument, from largest to smallest

    with open ('generated_file.txt','w') as gen_file:
        for key in size_dict.keys():
            gen_file.write(str(size_dict[key][0][0])+'\n')
            # print(size_dict[key][0][0])

def k_max_images_for_TIS(k, dir_in, dir_out):
    """ saves the k pictures with the highest GB size in the dir_out directory, for TIS project"""
    tisInfo = pd.DataFrame(columns=['image', 'score'])

    for folder in os.listdir(dir_in):
        print(folder)
        #loop over the subfolders and create a dictionary
        #{folder x1:[(file1,size1),(,)],folder x2:[(file2,size2),(,)]}
        getall = [ [files, os.path.getsize(dir_in + folder + "/" + files)] for files in os.listdir(dir_in + folder + "/") ]
        getall_dx = [x for x in getall if 'DX' in x[0]]
        sort_files = sorted(getall_dx, key=operator.itemgetter(1), reverse=True)
        for tile in sort_files[:k]:
            copyfile(dir_in + folder + '/' + tile[0],
                     dir_out + '/' + tile[0])
            tisInfo.loc[len(tisInfo)] = [tile[0], folder]
    tisInfo.to_csv('TISInfo25.csv')


def TIS_folder_names(directory_folders, directory_TIS, directory_excel):
    '''convert folder number to TIS score'''
    TIS_list={}
    with open(directory_excel) as fp:
        for line in fp:
            TIS = line.split(",")[1]
            line_list = line.split(".")
            patient = line_list[0]+"-"+line_list[1]+"-"+line_list[2]
            for filename in os.listdir(directory_folders):
                picture_list = os.listdir(directory_folders + "/" + filename)[0].split("-")  # takes the picture barcode
                picture = picture_list[0] + "-" + picture_list[1] + "-" + picture_list[2]  # cuts the patient barcode
                if patient == picture:
                    if TIS[0:5] not in TIS_list:
                        TIS_list[TIS[0:5]] = 1
                        shutil.copytree(directory_folders + filename, directory_TIS + TIS[0:5])   # create the folder
                    else:
                        for jpg in os.listdir(directory_folders+"/"+filename):
                            shutil.copy(directory_folders+"/"+filename+"/"+jpg, directory_TIS+"/"+TIS[0:5]+"/"+jpg)  #put in the exist folder
                    print(patient, TIS[0:5], filename)

def read_score(directory_folders_pic, directory_folders_scors, directory_csv_file):
    '''reads xml files and write the scores to the csv file
    directory_csv_file = The tis scores are written to this file. Must be initialized with headers: image (with the patches names below), and score. 
    directory_folders_scors = BRCA.txt file contain the image name and score for all images.
    directory_folders_pic = txt file of all the patches names that was tagged
    '''


    df = pd.read_csv(directory_csv_file)

    last_id = ''
    count = 0
    with open(directory_folders_scors) as scores, open(directory_folders_pic) as pictures:
        score_dic = {}
        print('making dictionary')
        for line in scores.readlines():
            line = line.rstrip('\n')
            line_list = line.split('\t')
            print(line_list)
            score_dic[line_list[0][:12]] = line_list[1]

        for line in pictures.readlines():
            print(count)
            id = line[:12]
            id = id.replace('-', '.')
            if id==last_id:
                df.at[count, 'score'] = score
            else:
                if id in score_dic:
                    score = score_dic[id]
                    df.at[count, 'score'] = score
            last_id = id
            count +=1

            

        df.to_csv ('/home/dsi/zurkin/export_dataframe.csv',index = False, header=True)

def svs_to_folders(directory_svs, directory_folders,directory_BRCA):
    count =0
    with open(directory_BRCA) as BRCA:
        BRCA_list = BRCA.readlines()
        for svs in os.listdir(directory_svs):
            count+=1
            if count<510:
                continue
            if svs=='TCGA-AO-A128-01Z-00-DX1.4E6BFFBC-87AD-4ED4-959D-FEB5545400BE.svs':
                shutil.copy(directory_svs + svs, directory_folders + "9/")
                continue
            svs_list = svs.split('-')
            patient = svs_list[0]+"."+svs_list[1]+"."+svs_list[2]
            for line in BRCA_list:
                fleg =0
                line = line.rstrip('\n')
                line = line.split("\t")
                if patient in line[0]:
                    fleg = 1
                    folder_num = int(float(line[1]))
                    shutil.copy(directory_svs + svs, directory_folders + str(folder_num) + "/")
                    print(count, patient, folder_num)
                    break
            if fleg==0:
                shutil.copy(directory_svs + svs, directory_folders + 'x/')

def take_relevant_patch(directory_patch, directory_out):
    for patch in os.listdir(directory_patch):
        patch_list = patch.split('.')
        pic_num = patch_list[2].split('-')
        count = int(pic_num[1])
        if count <= 25:
            shutil.copy(directory_patch + patch, directory_out + patch[:-3] + 'X.jpg')


base_dir = '/home/dsi/zurkin/data27/'


def merge_group(x):
    img = {}
    img_new = Image.new('RGB', (4000, 4000), (255, 255, 255))
    i = 0
    for ind, row in x.iterrows():
        img[i] = Image.open(base_dir+'all/'+row[0])
        if i >= 16:
            break
        i += 1
    if i < 15:
        return
    for i in range(4):
        for j in range(4):
            img_new.paste(img[i*4+j], (1000*i, 1000*j))
    img_new.save(base_dir+'large/'+row['image_name']+'.jpg', 'JPEG')


def merge_images():
    df = pd.read_csv(base_dir+'1045.csv')
    df['image_name'] = df.image.apply(lambda x: x[5:12])
    df.groupby('image_name').apply(merge_group)


if __name__ == "__main__":
    merge_images()
    #k_max_images_for_TIS(25, '/home/zel/rimon/Ray/bin/TIS/input_TIS/', '/home/zel/rimon/Ray/bin/TIS/input_TIS25/')

    #read_score('/home/dsi/zurkin/data15_test/test.txt', '/home/dsi/zurkin/BRCA.txt', '/home/dsi/zurkin/Book1.csv')

    #take_relevant_patch('/home/zel/rimon/Ray/bin/TIS/tagged/FFPE_250/9/','/home/zel/rimon/Ray/bin/TIS/tagged/FFPE_250/only25/')

    #k_max_images_to_csv('/home/zel/rimon/Ray/bin/TIS/tagged/tejal_images/images_776_1000/')

    #drop_background(200, 20, 'TCGA-OL-A66O-01Z-00-DX1.5F1E4C60-5CE8-41B4-A94D-4AA80D9253F9.92-20.jpg', '/home/zel/rimon/Ray/bin/TIS/tagged/tejal_images/images_2/', '/home/zel/rimon/Ray/bin/TIS/data12/')
