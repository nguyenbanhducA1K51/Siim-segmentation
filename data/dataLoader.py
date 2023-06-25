import json
import glob
from easydict import EasyDict as edict
import torchvision.transforms as T
from albumentations.pytorch import ToTensorV2
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
import sys
sys.path.append("../data")


class SiimDatast(Dataset):
    def __init__(self, cfg, csv_file, transform=None, sampleRate=1, mode="train"):
        self.cfg = cfg
        self.rootImage = cfg.path.pngImages
        self.rootMask = cfg.path.pngMasks
        self.csv = csv_file
        self.transform = transform
       
        self.df = pd.read_csv(csv_file)
        # self.df.info()
        numPos = self.df[self.df["has_pneumo"] == 1].shape[0]
        numNeg = self.df[self.df["has_pneumo"] == 0].shape[0]
        if cfg.data.useMiniData == "yes":
            if mode == "train":
                numPos = min(numPos, cfg.data.miniData.train)
            else:
                numPos = min(numPos, cfg.data.miniData.val)

        self.posDf = self.df[self.df["has_pneumo"] == 1].sample(n=numPos)
        # print (self.posDf.shape[0])
        self.negDf = self.df[self.df["has_pneumo"] == 0]
        self.lenTotal = len(self.df)
        sampleNeg = self.negDf.sample(n=self.posDf.shape[0]*sampleRate)
        # fraction parameter specifies the fraction of rows to be sampled, and it should be a float between 0 and 1.
        self.balanceDf = self.posDf.append(
            sampleNeg).sample(frac=1).reset_index(drop=True)
       
    def __len__(self):
        return self.balanceDf.shape[0]

    def __getitem__(self, idx):
        
        item = self.balanceDf.iloc[idx]
        imgPathBase = item.loc["new_filename"]
        imgId = item.loc["ImageId"]
    
        mask = item["has_pneumo"]

        imgPath = self.rootImage+"/"+str(imgPathBase)
        maskPath = self.rootMask+"/"+str(imgPathBase)
        
        image = cv2.imread(imgPath,0)
        mask = cv2.imread(maskPath,0)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
        return transformed["image"], transformed["mask"]
    
    def visualize(self,idx):
         img,mask=self.__getitem__(idx)
         fig,ax=plt.subplots(3)
         print (img.shape)
         print (mask.shape)
    
         ax[0].imshow(img[0,:,:],cmap="gray")
         ax[1].imshow(img.reshape(320,320,3))
         ax[2].imshow(mask,cmap="gray")
         plt.show()




def norm (image,*arg,**karg):
            image= np.divide(np.array(image,copy=True).astype(np.float32),255)
            image= np.expand_dims(image, axis=0)
            return np.repeat(image,3,axis=0)

def loadData(cfg, mode="default"):
    batch_size = cfg.train.batch_size
    trainTransform = A.Compose([
        A.Resize(height=340,width=340),
        A.RandomSizedCrop(min_max_height=(300,330),height=320,width=320,w2h_ratio=1.0),
         A.Lambda( image=norm),

    ])
    valTransform = A.Compose([
        A.Resize(height=320,width=320),

    ])
    trainset = SiimDatast(cfg=cfg,csv_file=cfg.path.trainCsv, transform=trainTransform, mode="train")
    testset = SiimDatast(cfg=cfg,csv_file=cfg.path.testCsv, transform=valTransform, mode="test")
    img,mask=trainset.__getitem__(1)
    trainset.visualize(3)
    print (np.unique(mask))
    trainLoader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True)
    testLoader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False)
    return trainLoader,testLoader




cfg_path="../config/config.json" 
with open(cfg_path) as f:
    cfg = edict(json.load(f))
# numclass,train_loader,test_loader=loadData(cfg=cfg)

path= "/Users/mac/Downloads/siim-acr-pneumothorax/png_images/0_train_0_.png"

train,test=loadData(cfg=cfg)


