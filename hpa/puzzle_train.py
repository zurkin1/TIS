# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import math
import shutil
import random
import argparse
import numpy as np
import time
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader, Dataset

from puzzle_utils import *
from puzzle_networks import *


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../../data/public/', type=str)
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--max_epoch', default=15, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--wd', default=0.01, type=float)
parser.add_argument('--nesterov', default=True, type=int)
parser.add_argument('--image_size', default=512, type=int)
# For Puzzle-CAM
parser.add_argument('--num_pieces', default=4, type=int)
# 'cl_pcl' # 'cl_re' # 'cl_conf' # 'cl_pcl_re' # 'cl_pcl_re_conf'
parser.add_argument('--loss_option', default='cl_pcl_re', type=str)
parser.add_argument('--level', default='feature', type=str)
parser.add_argument('--re_loss_option', default='masking', type=str) # 'none', 'masking', 'selection'
parser.add_argument('--alpha', default=1.0, type=float)
parser.add_argument('--alpha_schedule', default=0.50, type=float)
parser.add_argument('--num_workers', default=64, type=int) #128

device = torch.device(f'cuda:0' if torch.cuda.is_available() else "cpu")
device_ids = [0]


class Average_Meter:
    def __init__(self, keys):
        self.keys = keys
        self.clear()

    def add(self, dic):
        for key, value in dic.items():
            self.data_dic[key].append(value)

    def get(self, keys=None, clear=False):
        if keys is None:
            keys = self.keys

        dataset = [float(np.nanmean(self.data_dic[key])) for key in keys]
        if clear:
            self.clear()

        if len(dataset) == 1:
            dataset = dataset[0]

        return dataset

    def clear(self):
        self.data_dic = {key : [] for key in self.keys}


class PolyOptimizer(torch.optim.SGD):
    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9, nesterov=False):
        super().__init__(params, lr, weight_decay, nesterov=nesterov)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):
        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1


class HpaDataset(Dataset):
    def __init__(self, root, phase, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(root + '../' + csv_file)
        ind = int(0.9 * len(self.df))
        if phase == 'train':
            self.df = self.df.iloc[0:ind]
        else:
            self.df = self.df.iloc[ind:]
            self.df.reset_index(inplace=True, drop=True)
        #self.df = self.df.loc[self.df.group == phase]
        #self.df.reset_index(inplace=True, drop=True)
        self.root_dir = root # + phase
        self.transform = transform
        self.phase = phase
        self.lb = MultiLabelBinarizer().fit([range(0,19)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.df.ID[idx])
        image = Image.open(img_name +  '.png')
        if self.transform:
            image = self.transform(image)
        label = self.df.Label[idx] #.as_matrix().astype('float')
        labels = [int(x) for x in label.split('|')]
        label = self.lb.transform([labels])

        return image, label


if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()
    log_func = lambda string='': print(string)

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    _transform = {}
    _transform['train'] = transforms.Compose([
            transforms.RandomCrop(size=(256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ]
    )
    _transform['validate'] = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    #train_dataset = VOC_Dataset_For_Classification(args.data_dir, 'train_aug', train_transform)
    _dataset = {}
    _loader = {}

    for phase in ['train', 'validate']:
        _dataset[phase] = HpaDataset(args.data_dir, phase, 'kaggle2.csv', _transform[phase])
        _loader[phase] = DataLoader(_dataset[phase], batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)

    train_iteration = len(_loader['train'])
    max_iteration = args.max_epoch * train_iteration

    ###################################################################################
    # Network
    ###################################################################################
    model = Classifier(args.architecture)
    #param_groups = model.get_parameter_groups(print_fn=None)
    gap_fn = model.global_average_pooling_2d
    #model = nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)

    #log_func('[i] Total Params: %.2fM'%(calculate_parameters(model)))
    #log_func('[i] The number of pretrained weights : {}'.format(len(param_groups[0])))
    #log_func('[i] The number of pretrained bias : {}'.format(len(param_groups[1])))
    #log_func('[i] The number of scratched weights : {}'.format(len(param_groups[2])))
    #log_func('[i] The number of scratched bias : {}'.format(len(param_groups[3])))

    ###################################################################################
    # Loss, Optimizer
    ###################################################################################
    class_loss_fn = nn.MultiLabelSoftMarginLoss(reduction='none').to(device)
    re_loss_fn = nn.L1Loss().to(device)

    #optimizer = PolyOptimizer([
    #    {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
    #    {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
    #    {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wd},
    #    {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0},
    #], lr=args.lr, momentum=0.9, weight_decay=args.wd, max_step=max_iteration, nesterov=args.nesterov)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    #################################################################################################
    # Train
    #################################################################################################
    train_meter = Average_Meter(['loss', 'class_loss', 'p_class_loss', 're_loss', 'alpha'])
    #writer = SummaryWriter(tensorboard_dir)
    loss_option = args.loss_option.split('_')
    best_loss = 1000

    for epoch in range(args.max_epoch):
        # Each epoch has a training and validation phase
        iteration = epoch * train_iteration
        for phase in ['train', 'validate']:
            log_func('Epoch {}/{} {}'.format(epoch, args.max_epoch - 1, phase))
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            # Iterate over data.
            for batch_idx, (images, labels) in enumerate(tqdm(_loader[phase])): #enumerate(tqdm())
                #images, labels = train_iterator.get()
                images, labels = images.cuda(), labels.cuda()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if epoch <= 3:
                        for param in model.parameters():
                            param.requires_grad = False
                        for param in model.classifier.parameters(): #module.
                            param.requires_grad = True
                    else:
                        for param in model.parameters():
                            param.requires_grad = True
                    ###############################################################################
                    # Normal
                    ###############################################################################
                    logits, features = model(images, with_cam=True)

                    ###############################################################################
                    # Puzzle Module
                    ###############################################################################
                    tiled_images = tile_features(images, args.num_pieces)
                    tiled_logits, tiled_features = model(tiled_images, with_cam=True)
                    re_features = merge_features(tiled_features, args.num_pieces, args.batch_size)

                    ###############################################################################
                    # Losses
                    ###############################################################################
                    if args.level == 'cam':
                        features = make_cam(features)
                        re_features = make_cam(re_features)

                    class_loss = class_loss_fn(logits, labels).mean()

                    if 'pcl' in loss_option:
                        p_class_loss = class_loss_fn(gap_fn(re_features), labels).mean()
                    else:
                        p_class_loss = torch.zeros(1).cuda()

                    if 're' in loss_option:
                        if args.re_loss_option == 'masking':
                            class_mask = labels.unsqueeze(2).unsqueeze(3)
                            re_loss = re_loss_fn(features, re_features) * class_mask
                            re_loss = re_loss.mean()
                        elif args.re_loss_option == 'selection':
                            re_loss = 0.
                            for b_index in range(labels.size()[0]):
                                class_indices = labels[b_index].nonzero(as_tuple=True)
                                selected_features = features[b_index][class_indices]
                                selected_re_features = re_features[b_index][class_indices]

                                re_loss_per_feature = re_loss_fn(selected_features, selected_re_features).mean()
                                re_loss += re_loss_per_feature
                            re_loss /= labels.size()[0]
                        else:
                            re_loss = re_loss_fn(features, re_features).mean()
                    else:
                        re_loss = torch.zeros(1).cuda()

                    #if 'conf' in loss_option:
                    #    conf_loss = shannon_entropy_loss(tiled_logits)
                    #else:
                    #    conf_loss = torch.zeros(1).cuda()

                    if args.alpha_schedule == 0.0:
                        alpha = args.alpha
                    else:
                        alpha = min(args.alpha * iteration / (max_iteration * args.alpha_schedule), args.alpha)

                    loss = class_loss + p_class_loss + alpha * re_loss

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                #################################################################################################
                # Statistics.
                #################################################################################################
                train_meter.add({
                    'loss' : loss.item(),
                    'class_loss' : class_loss.item(),
                    'p_class_loss' : p_class_loss.item(),
                    're_loss' : re_loss.item(),
                    'alpha' : alpha,
                })

            loss, class_loss, p_class_loss, re_loss, alpha = train_meter.get(clear=True)
            learning_rate = float(get_learning_rate_from_optimizer(optimizer))
            log_func(f'[i] \
                learning_rate={learning_rate:.6f} \
                loss={loss:.6f} \
                class_loss={class_loss:.6f} \
                p_class_loss={p_class_loss:.6f} \
                re_loss={re_loss:.6f}'
            ) #alpha={alpha:.2f}, conf_loss={conf_loss:.4f}
            #writer.add_scalar('Train/loss', loss, iteration)
            #writer.add_scalar('Train/class_loss', class_loss, iteration)
            #writer.add_scalar('Train/p_class_loss', p_class_loss, iteration)
            #writer.add_scalar('Train/re_loss', re_loss, iteration)
            #writer.add_scalar('Train/conf_loss', conf_loss, iteration)
            #writer.add_scalar('Train/learning_rate', learning_rate, iteration)
            #writer.add_scalar('Train/alpha', alpha, iteration)

            #################################################################################################
            # Evaluation
            #################################################################################################
            if phase == 'validate' and loss < best_loss:
                best_loss = loss
                torch.save(model, 'baselinepc')
                log_func('[i] save model')

    #writer.close()
