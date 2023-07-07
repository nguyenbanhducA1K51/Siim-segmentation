import json
import glob
from easydict import EasyDict as edict
import torchvision.transforms as T
import albumentations as A
from torch.utils.data import random_split
from torch.nn import functional as F
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import random
from torch.utils.data.sampler import Sampler
class SiimDatast(Dataset):
    def __init__(self, cfg, csv_file, transform=None,  mode="train"):
        self.cfg = cfg
        self.rootImage = cfg.path.pngImages
        self.rootMask = cfg.path.pngMasks
        self.csv = csv_file
        self.transform = transform
        self.mode=mode
        self.df = pd.read_csv(csv_file)
        print (" Total number of sample in original {}set : {}".format( mode,len(self.df)))
         
        self.df["orinIndex"]=self.df.index     
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        
        item = self.df.iloc[idx]
        imgPathBase = item.loc["new_filename"]
        imgId = item.loc["ImageId"]
        oriIndex=self.df.iloc[idx].loc["orinIndex"]

        imgPath = self.rootImage+"/"+str(imgPathBase)
        maskPath = self.rootMask+"/"+str(imgPathBase)
        # image is already in range (0,1)
        image = cv2.imread(imgPath,0)
        mask = cv2.imread(maskPath,0)  
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image=transformed["image"]
            mask=transformed["mask"]
        
        return oriIndex, image,mask
    
    def getsampler(self):
            if self.cfg.sampler.type=="ratio":                
                return  PneumoSampler(cfg=self.cfg,df=self.df,mode=self.mode)
                
            elif self.cfg.sampler.type=="random":              
                return    RandomSampler(cfg=self.cfg,mode=self.mode,df=self.df)
            else:
                return None
def loadData(cfg, mode="default"):
    def norm (mask,*args,**kargs):
        mask=mask/255
        mask.astype(float) 
        return mask
    def stackChannel(image,*args,**kargs):
        image=np.expand_dims(image,axis=0)
        return np.repeat(image,3,axis=0)
    trainTransform = A.Compose([
        A.ShiftScaleRotate( rotate_limit=15),
        A.HorizontalFlip(),
        A.Resize(height=512,width=512),   
        A.RandomSizedCrop(min_max_height=( 500,510),height=512,width=512),
        A.Lambda(image=stackChannel, mask=norm),
       
    ])
    valTransform = A.Compose([

        A.Resize(height=512,width=512),
        A.Lambda(image=stackChannel,mask=norm),
       
        
    ])
    trainset = SiimDatast(cfg=cfg,csv_file=cfg.path.trainCsv,transform=trainTransform , mode="train")
    trainsampler=trainset.getsampler()
    testset = SiimDatast(cfg=cfg,csv_file=cfg.path.testCsv, transform=valTransform, mode="test")
    valsampler=testset.getsampler()
    trainLoader = torch.utils.data.DataLoader(
        trainset,num_workers=8,pin_memory=True, batch_size=cfg.train.batch_size , sampler=trainsampler if trainsampler is not None else None, shuffle=True if trainsampler is  None else False)
    testLoader = torch.utils.data.DataLoader(
        testset, num_workers=8,batch_size=1,pin_memory=True,sampler=valsampler )
    return trainLoader,testLoader
def tta_testLoader(cfg):
    df = pd.read_csv(cfg.path.testCsv)
    sampler=RandomSampler(cfg=cfg,mode="tta eval ",df=df)
    def norm (mask,*args,**kargs):
        mask=mask/255
        mask.astype(float) 
        return mask
    def stackChannel(image,*args,**kargs):
        image=np.expand_dims(image,axis=0)
        return np.repeat(image,3,axis=0)
    val_transform = A.Compose([
        A.ShiftScaleRotate( rotate_limit=15),
        A.HorizontalFlip(),
        A.Resize(height=512,width=512),   
        A.RandomSizedCrop(min_max_height=( 500,510),height=512,width=512),
        A.Lambda(image=stackChannel, mask=norm),
       
    ])
    testset = SiimDatast(cfg=cfg,csv_file=cfg.path.testCsv, transform=val_transform, mode="test")
    test_loader = torch.utils.data.DataLoader(
        testset, num_workers=8,batch_size=1,pin_memory=True,sampler=sampler)
    return test_loader

class PneumoSampler(Sampler):
    def __init__(self, cfg,df,mode, positive_perc=0.8):
        self.df = df
        self.positive_perc = positive_perc
        self.positive_idxs = self.df.query('has_pneumo==1').index.values
        self.negative_idxs = self.df.query('has_pneumo!=1').index.values
        self.n_positive = len(self.positive_idxs)     
        self.n_negative = int(self.n_positive * (1 - self.positive_perc) / self.positive_perc)
        print("total sample in real sampler :{}".format(self.n_positive + self.n_negative))
    def __iter__(self):
        negative_sample = np.random.choice(self.negative_idxs, size=self.n_negative)
        positive_sample=np.random.choice(self.positive_idxs, size=self.n_positive)
        shuffled = np.random.permutation(np.hstack((negative_sample, positive_sample)))
        return iter(shuffled.tolist())

    def __len__(self):
        return self.n_positive + self.n_negative

class RandomSampler(Sampler):
    def __init__(self,cfg,mode,df):
       
        self.cfg=cfg
        self.samples=min(cfg.sampler.randomSampler.number,len(df))
        self.df=df
        print (" Total random samples {}: ".format(self.samples))
    def __len__(self):
        return self.samples
    def __iter__(self):
        shuffle=np.random.choice(np.arange(0,len(self.df)),size=self.samples)
        return iter(shuffle)









