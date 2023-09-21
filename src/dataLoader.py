import json
import glob
from easydict import EasyDict as edict
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
import random
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from typing import Literal,List
import sys
sys.path.append("/root/repo/Siim-segmentation")
sys.path.append("..")
from src.utils import load_transform, unique_to_list,rle2mask
import pydicom
from torch.utils.data import WeightedRandomSampler, RandomSampler

class SIIMDataset(Dataset):
    def __init__(self,cfg,mode:Literal["train","val","test"],fold=1):
        self.cfg=cfg
        self.mode=mode
        if mode=="train" or mode=="val":
            self.df=pd.read_csv(os.path.join(cfg.path.png_save_path,"k_fold.csv"))
            self.df["fold"]=self.df["fold"].astype(int)

            if mode=="train":
                
                self.df=self.df[self.df['fold']!=int(fold)].copy().reset_index(drop=True)
            else:
                self.df=self.df[self.df['fold']==int(fold)].reset_index(drop=True)
            
            if self.cfg.train.mini_data.train==-1:
                mini_data=len(self.df)
            else:
                mini_data=min( self.cfg.train.mini_data.train,len(self.df))
            self.df=self.df[:mini_data]
        else:
            self.df=pd.read_csv(os.path.join(cfg.path.png_save_path,"test.csv"))

        self.transform=load_transform(cfg,mode=mode)

    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        if self.mode=="train" or self.mode=="val":
            img_path=os.path.join(self.cfg.path.png_save_path,"img",f"{self.df.iloc[idx]['ImageId']}.png")
            msk_path=os.path.join(self.cfg.path.png_save_path,"msk",f"{self.df.iloc[idx]['ImageId']}.png")
        
        image=cv2.imread(img_path,0)
        mask=cv2.imread(msk_path,0)
        transformed = self.transform(image=image, mask=mask)
        image=transformed["image"]
        mask=transformed["mask"]
        return image,mask

 
    def check_mask_integrity(self):
   
        n_samples=30
        dataset_idx=random.sample(range(self.__len__()),n_samples)
        
        ## check after go through transform
        def condition (x):
            msk_path=os.path.join(self.cfg.path.png_save_path,"msk",f"{self.df.iloc[x]['ImageId']}.png")
            mask=cv2.imread(msk_path,0)
            if np.sum(mask)>0:
                print (f"mask before transform {np.unique(mask)}")
                return True
            return False
        
        positive_idx=[x for x in dataset_idx if condition(x)]

        def condition (x):
            img,msk= self.__getitem__(x)
            if np.sum(msk)>0:
                print (f"mask unique {np.unique(msk)}")
                return True
            return False
        
        positive_idx=[x for x in dataset_idx if condition(x)]


class UpSampler(Sampler):
    def __init__(self, dataset,total_epoch=1):
        self.upper=0.8
        self.lower=0.3

        self.dataset = dataset
        self.num_samples = len(dataset)
       
        self.total_epoch=total_epoch
        self.cur_epoch=0
        self.df=dataset.df
        self.pos_idx=self.df[self.df["has_pneumo"]==1].index.tolist()
        self.neg_idx=self.df[self.df["has_pneumo"]!=1].index.tolist()
        self.set_epoch(0)

    def set_epoch(self,epoch):

        self.cur_epoch=epoch
        self.pos_ratio= self.upper-(self.upper-self.lower)*(epoch/ (self.total_epoch-1))
        num_negs_samples= int ( len(self.pos_idx)*(1/self.pos_ratio-1))
        self.neg_samples= random.sample(self.neg_idx, num_negs_samples)
        
        self.indices=self.pos_idx+self.neg_samples
        random.shuffle(self.indices)
    def __iter__(self):
       
        return iter(self.indices)

    def __len__(self):
        return len (self.neg_samples) + len(self.pos_idx)
    
def visualize (dataset):
    n_samples=20
    dataset_idx=random.sample(range(len(dataset)),n_samples)
    def condition (x):
        img,msk= dataset.__getitem__(x)
        if np.sum(msk)>0:
            return True
        return False

    positive_idx=[x for x in dataset_idx if condition(x)]
    save_path="/root/repo/Siim-segmentation/config/img_msk_visualize.png"
    if len (positive_idx)==0:
        print ("All are negative samples")
    else:
        fig, ax=plt.subplots( len(positive_idx),9, figsize=(30,30))
        plt.tight_layout()
        for i, idx in enumerate(positive_idx):
            img,msk=dataset.__getitem__(idx)
            

            ax[i,0].imshow(img[0],cmap="bone")
            if np.sum(msk)>0:
                ax[i,0].set_title(f" has pneumo { np.max(msk)}")
            ax[i,1].imshow(msk,cmap="bone")
            ax[i,2].imshow(img[0],cmap="bone")
            ax[i,2].imshow(msk,alpha=0.5, cmap='Reds')
            img,msk=dataset.__getitem__(idx)
            ax[i,3].imshow(img[0],cmap="bone")
            ax[i,4].imshow(msk,cmap="bone")
            ax[i,5].imshow(img[0],cmap="bone")
            ax[i,5].imshow(msk,alpha=0.5, cmap='Reds')
            img,msk=dataset.__getitem__(idx)
            ax[i,6].imshow(img[0],cmap="bone")
            ax[i,7].imshow(msk,cmap="bone")
            ax[i,8].imshow(img[0],cmap="bone")
            ax[i,8].imshow(msk,alpha=0.5, cmap='Reds')

        plt.show()
        plt.savefig(save_path)
        print ("FINISH VISUALIZING")

if __name__=="__main__":
    from easydict import EasyDict as edict
    import os,json
    cfg_path="/root/repo/Siim-segmentation/config/config.json"
    with open(cfg_path) as f:
        cfg = edict(json.load(f))
    
    trainset=SIIMDataset(cfg,mode="train")
    visualize(trainset)
    # print (len(trainset))

    # sampler=UpSampler(trainset,10)
    # loader=DataLoader(trainset, sampler=sampler, batch_size=10)
    
    
        







