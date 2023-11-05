from __future__ import print_function, division
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import shutil
import copy
import imageio
from skimage.io import imread
from tqdm import tqdm
import sys
import imageio
from PIL import Image
from sklearn.metrics import roc_auc_score
import cv2
import pylab
from scipy import ndimage
import subprocess
import random


batch_size=32 # 64
root_path = '/home/dsi/zurkin/data23/'
image_res=500
device = torch.device(f'cuda:{3}' if torch.cuda.is_available() else "cpu")


def auto_gpu_selection(usage_max=0.01, mem_max=0.7):
    """Auto set CUDA_VISIBLE_DEVICES for gpu
    :param mem_max: max percentage of GPU utility.
    :param usage_max: max percentage of GPU memory.
    :return: gpu number.
    """
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    log = str(subprocess.check_output("nvidia-smi", shell=True)).split(r"\n")[6:-1]
    gpu = 0

    # Maximum of GPUS, 8 is enough for most
    for i in range(8):
        idx = i*3 + 2
        if idx > log.__len__()-1:
            break
        inf = log[idx].split("|")
        if inf.__len__() < 3:
            break
        usage = int(inf[3].split("%")[0].strip())
        mem_now = int(str(inf[2].split("/")[0]).strip()[:-3])
        mem_all = int(str(inf[2].split("/")[1]).strip()[:-3])
        print("GPU-%d : Usage:[%d%%]" % (gpu, usage))
        if mem_now < mem_max*mem_all: # and usage < 100*usage_max:
            os.environ["CUDA_VISIBLE_EVICES"] = str(gpu)
            print("\nAuto choosing vacant GPU-%d : Memory:[%dMiB/%dMiB] , GPU-Util:[%d%%]\n" %
                  (gpu, mem_now, mem_all, usage))
            return gpu
        print("GPU-%d is busy: Memory:[%dMiB/%dMiB] , GPU-Util:[%d%%]" %
              (gpu, mem_now, mem_all, usage))
        gpu += 1
    print("\nNo vacant GPU, use CPU instead\n")
    os.environ["CUDA_VISIBLE_EVICES"] = "-1"


def particle_count():
    df1 = pd.read_csv(root_path+'data_v1c.csv')
    #df['pcount'] = df.image.apply(lambda x: count_particles(x))
    start = int(sys.argv[1])
    for ind, row in df1.iloc[start:start+5000].iterrows():
        im = cv2.imread(root_path+'all/'+row[0]) # TCGA-E9-A1RF-01Z-00-DX1.ee37dfd4-1431-4e80-bd7e-3e1f878273e0.954-12.jpg')
        """
        pylab.figure(0)
        pylab.imshow(im)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        maxValue = 255
        adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C #cv2.ADAPTIVE_THRESH_MEAN_Ci #cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        thresholdType = cv2.THRESH_BINARY#cv2.THRESH_BINARY #cv2.THRESH_BINARY_INV
        blockSize = 5 #odd number like 3,5,7,9,11
        C = -3 # constant to be subtracted
        im_thresholded = cv2.adaptiveThreshold(gray, maxValue, adaptiveMethod, thresholdType, blockSize, C)
        im = ndimage.gaussian_filter(im, sigma=1/(4.*10))
        """
        im = ndimage.median_filter(im, 3)
        mask = im > im.mean()
        labelarray, particle_count = ndimage.measurements.label(mask) # im_thresholded)
        #pylab.figure(1)
        #pylab.imshow(im) # im_thresholded)
        #pylab.show()
        #print(particle_count)
        df1.loc[ind, 'pcount'] = particle_count
        if ind % 10 == 0:
            print('\r', ind, end="", flush=True)

    df2 = pd.read_csv(root_path+'data_v1c.csv')
    df1['pcount'] = df1.pcount + df2.pcount
    df1.to_csv(root_path+'data_v1c.csv', index=False)
    

def prepare_data(csv_file='1045.csv'):
    df = pd.read_csv(root_path+csv_file)
    #df['int_score'] = df.score.round()
    #df_5 = df.loc[df.int_score == 5].sample(4300).copy()
    #df = df.loc[(df.int_score == 6) & (df.group == 'train')].copy()
    #df = pd.concat([df_5, df_not_5])
    #df = pd.read_csv('/gdrive/My Drive/TIS/public/set_1_2_3_1000.csv')[['image', 'score']]
    #(train, test) = train_test_split(df, test_size=0.25, random_state=42)
    #df['bin_score'] = df.score.apply(lambda x: to_categorical(0 if x < 7 else 1, num_classes=2))
    df['bin_score'] = df.score.apply(lambda x: 'low' if x < 5.6 else 'high')
    df['size'] = 0


    df['group'] = 'nogroup'
    df['image_name'] = df.image.apply(lambda x: x[5:12])
    df = df.sample(frac=1)
    image_num = df.image_name.unique()
    train, validate, test = np.split(image_num, [int(.70*len(image_num)), int(.85*len(image_num))])
    df.loc[df.image_name.isin(train), 'group'] = 'train' #[['image', 'bin_score']].sample(frac=1)
    df.loc[df.image_name.isin(test), 'group'] = 'test' #[['image', 'bin_score']].sample(frac=1)
    df.loc[df.image_name.isin(validate), 'group'] = 'validate' #[['image', 'bin_score']].sample(frac=1)

    dest_dir = '/home/dsi/zurkin/data22'
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
        for d in ['/train/', '/validate/', '/test/']:
            os.mkdir(dest_dir+d)
            for l in ['low', 'high']:
                os.mkdir(dest_dir+d+l)
    
    #df2 = pd.read_csv('data3/data.csv')
    #df2['image_name'] = df2.image.apply(lambda x: x[5:12])
    #df2 = set(df2.image_name)
    #df2 = df.groupby(['image_name'], as_index=False).apply(lambda x: x if len(x)< i+1 else x.iloc[[i]]).reset_index(level=0, drop=True)
    df['size'] = df.image.apply(lambda x: os.path.getsize(root_path+x))
    df2 = df.copy().sort_values('size', ascending=False).drop_duplicates('image_name')
    #df.to_csv(root_path+'data45a.csv')

    for ind, row in df2.iterrows():
        if os.path.isfile(root_path+row['image']):
            shutil.copy(root_path+row['image'], f'{dest_dir}/{row.group}/{row.bin_score}/{str(round(row.score,2))}_{row.image}') # {row[5]}/{row[3]}/{row[0]}')

    #for ind, row in df.iterrows():
    #    if row[4] == 'test':
    #        shutil.copy(root_path+row[0], f'{dest_dir}_test_all/{row[2]}/{row[0]}')

    #df.to_csv(dest_dir+'/data_v1.csv')
    df2.to_csv(dest_dir+'/data_v2.csv')


def prepare_data2():
    source_path = '/home/dsi/zurkin/data/train/'
    dest_dir = '/home/dsi/zurkin/data/train_1/'
    DIRS = ["nucleoplasm", "nuclear_membrane", "nucleoli", "nucleoli_fibrillar_center", "nuclear_speckles",\
            "nuclear_bodies", "endoplasmic_reticulum", "golgi_apparatus", "intermediate_filaments", "actin_filaments",\
            "microtubules", "mitotic_spindle", "centrosome", "plasma_membrane", "mitochondria", "aggresome", "cytosol",\
            "vesicles", "negative"]
    for dir1 in DIRS:
        os.mkdir(dest_dir+dir1)
        X = [name for name in (os.listdir(source_path+dir1)) if '_1.png' in name]
        #sample_size=min(len(X), 500)
        #X = random.sample(X, sample_size)
        print(f'{dir1}: {len(X)}')
        for file in X:
            shutil.copy(f'{source_path}/{dir1}/{file}', f'{dest_dir}/{dir1}/{file}')
        #print(dirs)
        #print(files)


    #if not os.path.exists(dest_dir):
    #    os.mkdir(dest_dir)
    #    for d in ['/train/', '/validate/', '/test/']:
    #        os.mkdir(dest_dir+d)
    #        for l in ['low', 'high']:
    #            os.mkdir(dest_dir+d+l)
    
    #for ind, row in df2.iterrows():
    #    if os.path.isfile(root_path+row['image']):
    #        shutil.copy(root_path+row['image'], f'{dest_dir}/{row.group}/{row.bin_score}/{str(round(row.score,2))}_{row.image}') # {row[5]}/{row[3]}/{row[0]}')


class TisDataset(Dataset):
    def __init__(self, csv_file, root, phase, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(root + csv_file)
        self.df = self.df.loc[self.df.group == phase]
        #self.df = self.df.loc[(self.df.int_score < 5) | (self.df.int_score > 6)]
        self.df.reset_index(inplace=True, drop=True)
        self.root_dir = root # + phase
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.df.image[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        label = self.df.score[idx]/10 #.as_matrix().astype('float')
        #label = label.reshape(-1, 2)
        #sample = {'image': image, 'label': label}

        return image, label #sample


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(image_res), #224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validate': transforms.Compose([
        transforms.RandomResizedCrop(image_res), #256
        #transforms.CenterCrop(image_res),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.CenterCrop(image_res),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


def load_dataset(phase):
    #_dataset = TisDataset(root_path, phase, 'data_v2.csv', data_transforms[phase]) #torchvision.transforms.ToTensor()
    _dataset = torchvision.datasets.ImageFolder(os.path.join(root_path, phase), data_transforms[phase])
    _loader = torch.utils.data.DataLoader(_dataset, batch_size, num_workers=16, shuffle=True, pin_memory=True)
    return _loader


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.5 #1000000.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        if epoch <= 9:
        #    for param in model.parameters():
        #        param.requires_grad = True
            for param in model.classifier[9-epoch].parameters():
                param.requires_grad = True
        # Each epoch has a training and validation phase
        for phase in ['train', 'validate', 'test']:
            if (phase == 'test') and (epoch < (num_epochs - 1)):
                continue
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = torch.zeros(1, device=device)
            running_corrects = 0
            num_samples = 0

            # Iterate over data.
            for batch_idx, (inputs, labels) in enumerate(tqdm(load_dataset(phase))): #enumerate(tqdm())
                inputs = inputs.float()
                labels = labels.float()
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #if phase == 'train':
                    #    outputs = outputs.logits
                    #_, preds = torch.max(outputs, 1)
                    loss = criterion(outputs.view(inputs.size(0)), labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss #.item() #* inputs.size(0)
                #running_corrects += torch.sum(outputs.reshape(labels.shape[0]).sigmoid().round() == labels.round())
                running_corrects += torch.sum(outputs.reshape(labels.shape[0]).round() == labels.round())
                num_samples += labels.shape[0]
                #print("FOR TEST:","sum:",torch.sum(preds == labels).cpu().numpy(),"labels:", preds.cpu().numpy())
                
            #if phase == 'train':
            #    scheduler.step()

            epoch_acc = running_corrects.double() / num_samples
            avg_loss = running_loss.item() / (batch_idx + 1)
            print('{} loss: {:.4f}  acc: {:.4f}'.format(phase, avg_loss, epoch_acc))

            # deep copy the model
            if phase == 'validate' and epoch_acc > best_acc:
                best_acc = epoch_acc #avg_loss
                print('', file=open(root_path+f'/model_{str(round(float(epoch_acc), 4))}.sav', 'w'))
                #best_model_wts = copy.deepcopy(model.state_dict())
                print(f'Saving file.', flush=True)
                torch.save(model, root_path+f'/model.sav')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:4f}'.format(best_acc))
    # load best model weights
    #model.load_state_dict(best_model_wts)


#Visualize model predictions.
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(labels[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def predict():
    model = torch.load(root_path+'/model.sav', map_location=torch.device('cpu'))
    df = pd.read_csv(root_path+'/data_v2.csv')
    df = df.loc[df.group == 'test']
    df['pred'] = -1
    model = model.to('cpu')

    for ind, row in df.iterrows():
      myfile = root_path+f'/test/{row[3]}/{row[1]}'
      if os.path.isfile(myfile):
        image = Image.open(myfile)
        image = data_transforms['validate'](image)
        image = image.view(1,3,image_res,image_res)
        start = str(row[1]).find('TCGA')
        #df.loc[df.image == str(row[1])[start:], 'pred'] = float(model(image).sigmoid())
        df.loc[df.image == str(row[1])[start:], 'pred'] = float(model(image))
   
    df = df.loc[df.pred != -1]
    df['bin_score'] = df.bin_score.apply(lambda x: 0. if x == 'low' else 1.)
    #df['bin_pred'] = df.pred.apply(lambda x: 0 if x < 0.6 else 1)
    df2 = df.groupby('image_name').agg({'bin_score':np.max, 'pred':np.mean})
    print(roc_auc_score(df.bin_score, df.pred)) # , roc_auc_score(df2.bin_score, df2.pred))
    df.to_csv(root_path+'temp1.csv')

    
if __name__ == "__main__":
    #particle_count()
    #prepare_data()
    prepare_data2()

    #Resnet. 
    #model = models.resnet34(pretrained=True)
    #num_ftrs = model.fc.in_features
    #model.fc = nn.Linear(num_ftrs, 1)
    """    
    #Alexnet.
    model = models.alexnet(pretrained=True)
    #ft = list(model.features)
    #cl = list(model.classifier)
    #Remove the max pooling layer.
    #model.features = nn.Sequential(ft[0],ft[1],ft[2],ft[3],ft[4],ft[5],ft[6],ft[7],ft[8],ft[9],ft10],ft[11])
    #model.classifier = nn.Sequential(cl[0],cl[1],cl[2],cl[3],cl[4],cl[5],cl[6],nn.ReLU(inplace=True),nn.Linear(1000,1),nn.Sigmoid())
    #model.classifier[6] = nn.Linear(4096, 1)
    
    #Inception_v3.
    #model = models.inception_v3(pretrained=True)
    #num_ftrs = model.fc.in_features
    #model.fc = nn.Linear(num_ftrs, 1)  
    #for param in model.Mixed_7c.parameters(): # layer4
    #    param.requires_grad = True
    #for param in model.classifier.parameters():
    #    param.requires_grad = True
    
    #model = nn.Sequential(model, nn.Linear(1000,1))
    #model = torch.load(root_path+'/model.sav')
    #model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model = model.to(device)

    #For unbalanced classes:
    #Max(Number of occurrences in most common class) / (Number of occurrences in rare classes): low:200, high:10 -> weight = [200/200, 200/10]
    #weight = [1.6]
    #class_weight = torch.FloatTensor(weight).to(device)
    #criterion = nn.BCELoss() #pos_weight=class_weight)
    criterion = nn.MSELoss()
    optimizer_ft = optim.Adam(model.parameters(), lr=1e-6, weight_decay=0.01)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=100)
    predict()
    """