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
from typing import Literal,List
import sys
sys.path.append("/root/repo/Siim-segmentation/data")
from utils import load_transform, unique_to_list,rle2mask
import pydicom



class SIIMDataset(Dataset):
    def __init__(self,cfg,mode:Literal["train","val","test"]):
        if mode=="train":
            self.df=pd.read_csv(os.path.join(cfg.path.png_save_path,"train.csv"))
        elif mode=="val":
            self.df=pd.read_csv(os.path.join(cfg.path.png_save_path,"val.csv"))
        elif mode=="test":
            self.df=pd.read_csv(os.path.join(cfg.path.png_save_path,"test.csv"))

        self.transform=load_transform(cfg,mode=mode)
        self.cfg=cfg
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        img_path=os.path.join(cfg.path.png_save_path,"img",self.df.iloc[idx]['ImageId'].item())
        msk_path=os.path.join(cfg.path.png_save_path,"msk",self.df.iloc[idx]['ImageId'].item())
        img=cv2.imread(img_path,0)
        msk=cv2.imread(msk_path,0)
        transformed = self.transform(image=image, mask=mask)
        image=transformed["image"]
        mask=transformed["mask"]
        return image,mask

    def getsampler(self):
            if self.cfg.sampler.type=="ratio":                
                return  PneumoSampler(cfg=self.cfg,df=self.df,mode=self.mode)
                
            elif self.cfg.sampler.type=="random":              
                return    RandomSampler(cfg=self.cfg,mode=self.mode,df=self.df)
            else:
                return None
def visualize (img,msk):
    save_path="/root/repo/Siim-segmentation/data/img_msk_visualize.png"
    fig, ax=plt.subplots(1,3, fig_size=(30,30))
    plt.tight_layout()
    ax[0].imshow(img,cmap="bone")
    ax[1].imshow(msk,0)
    plt.show()
    plt.savefig(save_path)
    print ("FINISH VISUALIZING")
def generate_test_val_train_dataset(cfg,test_ratio=1/10,val_ratio=1/10):
    df=pd.read_csv(cfg.path.train_rle_csv)
    test_val_idx=random.sample(range(len(df)),int( (test_ratio+val_ratio)*len(df)))
    test_idx=test_val_idx[: int( (test_ratio)*len(df))]
    val_idx=test_val_idx[int( (test_ratio)*len(df)):]

    train_idx=[x for x in range(len(df)) if x not in  test_val_idx]

    test_df=df.iloc[test_idx].copy()
    val_df=df.iloc[val_idx].copy()
    train_df=df.iloc[train_idx].copy()
    testDataset=SIIMDataset(cfg,test_df,mode="test")
    trainDataset=SIIMDataset(cfg,train_df,mode='train')
    valDataset=SIIMDataset(cfg,val_df,mode="test")
    return trainDataset,testDataset,valDataset



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

class Tta():
    def __init__(self):
        return
    def get_tta_transform(self):
        transforms=[]
        transforms.append((forward_horiz,backward_horiz))
        transforms.append((forward_identity,backward_identity))
        return transforms

def forward_horiz(image):
    return image.flip(-1)
def backward_horiz(image):
    return image.flip(-1)
def forward_identity(image):
    return image
def backward_identity(image):
    return image
if __name__=="__main__":
    from easydict import EasyDict as edict
    import os,json
    cfg_path="/root/repo/Siim-segmentation/config/config.json"
    with open(cfg_path) as f:
        cfg = edict(json.load(f))
    trainset,testset,valset=generate_test_val_train_dataset(cfg=cfg)
    trainset.__getitem__(1)








