import os
import numpy as np
import random
import json

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from utils.tools import *


class CamObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, wavelet_gt_root, trainsize):
        
        self.trainsize = trainsize
        self.image_root = image_root
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.wavelet_gts = [wavelet_gt_root + f for f in os.listdir(wavelet_gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.wavelet_gts = sorted(self.wavelet_gts)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.wavelet_gt_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)), 
                                                transforms.ToTensor()])
        
    def __getitem__(self, index):
        image = rgb_loader(self.images[index])
        gt = binary_loader(self.gts[index])
        wavelet_gt = rgb_loader(self.images[index])     
        image, gt, wavelet_gt = self.aug_data(image=image, gt=gt, wavelet_gt=wavelet_gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        wavelet_gt = self.wavelet_gt_transform(wavelet_gt)  
        return image, gt, wavelet_gt
    
    def filter_files(self):
        assert len(self.images) == len(self.gts) == len(self.wavelet_gts)
        images = []
        gts = []
        wavelet_gts = []
        for img_path, gt_path, wavelet_gt_path in zip(self.images, self.gts, self.wavelet_gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            wavelet_gt = Image.open(wavelet_gt_path)
            if img.size == gt.size == wavelet_gt.size:
                images.append(img_path)
                gts.append(gt_path)
                wavelet_gts.append(wavelet_gt_path)
        self.images = images
        self.gts = gts
        self.wavelet_gts = wavelet_gts
    
    def resize(self, img, gt, wavelet_gt):
        assert img.size == gt.size == wavelet_gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), wavelet_gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt, wavelet_gt

    def __len__(self):
        return self.size   

    def aug_data(self, image, gt, wavelet_gt):
        image, gt, wavelet_gt = cv_random_flip(image, gt, wavelet_gt)
        image, gt, wavelet_gt = randomCrop(image, gt, wavelet_gt)
        image, gt, wavelet_gt = randomRotation(image, gt, wavelet_gt)
        image = colorEnhance(image)
        gt = randomPeper(gt)
        wavelet_gt = randomPeper(wavelet_gt)
        return image, gt, wavelet_gt

class test_dataset:
    """load test dataset (batchsize=1)"""
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = rgb_loader(self.images[self.index])
        image = self.transform(image)
        image = image.unsqueeze(0)
        gt = binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

class test_loader_faster(data.Dataset):
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.size = len(self.images)

    def __getitem__(self, index):
        images = self.rgb_loader(self.images[index])
        images = self.transform(images)
        img_name_list = self.images[index]

        return images, img_name_list

    def __len__(self):
        return self.size

def get_loader(image_root, gt_root, wavelet_gt_root, batchsize, trainsize, shuffle=True, num_workers=0, pin_memory=True):
    dataset = CamObjDataset(image_root, gt_root, wavelet_gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

    return data_loader                 