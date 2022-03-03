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

batch_size=256
root_path = '/home/dsi/zurkin/data17/all/'
image_res=500
device_id=6


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

def prepare_data(csv_file=root_path+'data_v4a.csv'):
    df = pd.read_csv(csv_file, index_col=0)
    #df['int_score'] = df.score.round()
    #df_5 = df.loc[df.int_score == 5].sample(4300).copy()
    #df = df.loc[(df.int_score == 6) & (df.group == 'train')].copy()
    #df = pd.concat([df_5, df_not_5])
    #df = pd.read_csv('/gdrive/My Drive/TIS/public/set_1_2_3_1000.csv')[['image', 'score']]
    #(train, test) = train_test_split(df, test_size=0.25, random_state=42)
    #df['bin_score'] = df.score.apply(lambda x: to_categorical(0 if x < 7 else 1, num_classes=2))
    df['bin_score'] = df.score.apply(lambda x: 'low' if x < 6. else 'high')
    #df['size'] = 0

    #df['group'] = 'nogroup'
    #df['image_name'] = df.image.apply(lambda x: x[5:12])
    #image_num = df.image_name.unique()
    #train, test = np.split(image_num, [int(.85*len(image_num))]) #, int(.85*len(image_num))])
    #df.loc[df.image_name.isin(train), 'group'] = 'train' #[['image', 'bin_score']].sample(frac=1)
    #df.loc[df.image_name.isin(test), 'group'] = 'test' #[['image', 'bin_score']].sample(frac=1)
    #df.loc[df.image_name.isin(validate), 'group'] = 'test' #[['image', 'bin_score']].sample(frac=1)
   
    #dest_dir = '/home/dsi/zurkin/data13/'
    #if not os.path.exists(dest_dir):
    #    os.mkdir(dest_dir)
    #    for d in ['train/', 'test/']:
    #        os.mkdir(dest_dir+d)
    #        for l in ['low', 'high']:
    #            os.mkdir(dest_dir+d+l)
    
    df2 = pd.read_csv('data3/data.csv')
    df2['image_name'] = df2.image.apply(lambda x: x[5:12])
    df2 = set(df2.image_name)
    #df2 = df.groupby(['image_name'], as_index=False).apply(lambda x: x if len(x)< i+1 else x.iloc[[i]]).reset_index(level=0, drop=True)
    for ind, row in df.iterrows():
        if not row[4] in df2:
            if os.path.isfile(root_path+row[0]):
                print('.', end="", flush=True)
        #if os.path.isfile(root_path+'train/1/'+row[0]):
    #       df['size'][ind] = os.path.getsize(root_path+row[0])
                shutil.copy(root_path+row[0], f'/home/dsi/zurkin/data14/{row[3]}/{row[0]}') # {row[5]}/{row[3]}/{row[0]}')
           #df['group'][ind] = 'train'
    #      shutil.move(f'/home/dsi/zurkin/data6/{row[0]}', f'/home/dsi/zurkin/data6/{row[4]}/{row[0]}')
    #df = df.sort_values('size', ascending=False).drop_duplicates('image_name')
    #df.to_csv('/home/dsi/zurkin/data7/1000/data_v4a.csv')

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(image_res), #224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        #transforms.RandomResizedCrop(image_res), #256
        transforms.CenterCrop(image_res),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def load_dataset(phase):
    _dataset = TisDataset(
        root=root_path,
        phase=phase,
        csv_file='data_v2.csv',
        transform=data_transforms[phase] #torchvision.transforms.ToTensor()
    )
    #_dataset = torchvision.datasets.ImageFolder(os.path.join('/home/dsi/zurkin/data2/', phase), data_transforms[phase])
    _loader = torch.utils.data.DataLoader(
        _dataset,
        batch_size=batch_size,
        num_workers=16,
        shuffle=True
    )
    return _loader


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 0.5 #1000000.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        #if epoch == 1:
        #   for param in model.parameters():
        #       param.requires_grad = True
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            num_samples = 0

            # Iterate over data.
            for batch_idx, (inputs, labels) in enumerate(tqdm(load_dataset(phase))): #enumerate(tqdm())
                #print('#', end="")
                #sys.stdout.flush()
                inputs = inputs.float()
                labels = labels.float()
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #_, preds = torch.max(outputs, 1)
                    loss = criterion(outputs.view(inputs.size(0)), labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() #* inputs.size(0)
                running_corrects += torch.sum(outputs.reshape(labels.shape[0]).sigmoid().round() == labels.round())
                num_samples += labels.shape[0]
                #print("FOR TEST:","sum:",torch.sum(preds == labels).cpu().numpy(),"labels:", preds.cpu().numpy())
                
            #if phase == 'train':
            #    scheduler.step()

            epoch_acc = running_corrects.double() / num_samples
            avg_loss = running_loss / batch_idx
            print('{} loss: {:.4f}  acc: {:.4f}'.format(phase, avg_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_loss: #avg_loss
                best_loss = epoch_acc #avg_loss
                print('', file=open(root_path+f'/model_{str(round(float(epoch_acc), 4))}.sav', 'w'))
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f'Saving file.')
                torch.save(model, root_path+f'/model.sav')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


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


def predict(model):
    df = pd.read_csv(root_path+'/data_v2.csv')
    df = df.loc[df.group == 'test']
    df['pred'] = -1
    model = model.to('cpu')

    for ind, row in df.iterrows():
      image = Image.open(root_path+'/'+row[1])
      image = data_transforms['test'](image)
      image = image.view(1,3,image_res,image_res)
      start = str(row[1]).find('TCGA')
      df.loc[df.image == str(row[1])[start:], 'pred'] = float(model(image).sigmoid())
   
    df = df.loc[df.pred != -1]
    df['bin_pred'] = df.pred.apply(lambda x: 0 if x < 0.6 else 1)
    df2 = df.groupby('image_name').agg({'bin_score':np.max, 'bin_pred':np.mean})
    print(roc_auc_score(df.bin_score, df.bin_pred), roc_auc_score(df2.bin_score, df2.bin_pred))

    
if __name__ == "__main__":
    prepare_data()
    """
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else "cpu")
    #Resnet. 
    #model = models.resnet18(pretrained=True)
    #num_ftrs = model.fc.in_features
    #model.fc = nn.Linear(num_ftrs, 1)
    #Alexnet.
    model = models.alexnet(pretrained=True)
    #ft = list(model.features)
    cl = list(model.classifier)
    #Remove the max pooling layer.
    #model.features = nn.Sequential(ft[0],ft[1],ft[2],ft[3],ft[4],ft[5],ft[6],ft[7],ft[8],ft[9],ft10],ft[11])
    model.classifier = nn.Sequential(cl[0],cl[1],cl[2],cl[3],cl[4],cl[5],cl[6],nn.ReLU(inplace=True),nn.Linear(1000,1))
    
    #model = nn.Sequential(model, nn.Softmax())
    #model = torch.load(root_path+'/model.sav')
    model = nn.DataParallel(model, device_ids=[6, 7])
    for param in model.parameters():
        param.requires_grad = True
    model = model.to(device)

    #for unbalanced classes:
    #Max(Number of occurrences in most common class) / (Number of occurrences in rare classes): low:200, high:10 -> weight = [200/200, 200/10]
    # weight = [603/348, 603/603]
    # class_weight = torch.FloatTensor(weight).to(device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight= class_weight)

    criterion = nn.BCEWithLogitsLoss() #MSELoss()
    optimizer_ft = optim.Adam(model.parameters()) #, lr=1e-5, weight_decay=0.01)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)
    predict(model)
    """
