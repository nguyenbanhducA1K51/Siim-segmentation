
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import tqdm
import sys
sys.path.append("../data")
from data import dataUtil
from . import modelUtils, models
from torch.optim import Adam
import torch.optim as optim
import segmentation_models_pytorch as smp
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

class SIIMnet():
    def __init__(self,cfg,device) :
        self.cfg=cfg
        self.device=device
        self.model=self.loadModel().to(self.device)
        self.optimizer,self.scheduler=self.loadOptimizer(model=self.model)
        self.criterion=self.loadCriterion()
        self.metric=modelUtils.metric
        
    def train_epoch(self,dataLoader,model,epoch):
        dices=modelUtils.AverageMeter()
        log_interval=self.cfg.train.log_interval
        model.train()
        with tqdm(dataLoader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            for idx,(oriIdx,x,y_true) in enumerate(tepoch) :

                self.optimizer.zero_grad()

                x=x.to(self.device).float()
                y_true=y_true.to(self.device).float()  
                y_hat=model(x).squeeze(1) 
                y_hat=y_hat.sigmoid()
                
                loss=self.criterion(y_hat,y_true)
                loss.backward()   
                self.optimizer.step()
                if idx % log_interval == 0:                   
                        tepoch.set_postfix(loss=loss.item())
        print ("-"*100)

    def eval (self,dataLoader,model,epoch):
        model.eval()
        losses=modelUtils.AverageMeter()
        metric=self.metric
        diceMetric=modelUtils.Metric()
        diceList=[]
        with torch.no_grad():
            with tqdm(dataLoader, unit="batch") as tepoch:
                for idx, (_,x,y_true) in enumerate(tepoch):
                    x=x.to(self.device).float()
                    y_hat=model(x).squeeze(1)     
                    y_true=y_true.to(self.device).float()
                    y_hat=torch.sigmoid(y_hat)
                    dice,term,denom=diceMetric.compDice(y_hat,y_true)  
                    diceList.append(dice)      
                    if idx%4==0:
                        tepoch.set_postfix(curdice=dice.item(),term=term.item(),denom=denom.item())
                meandice=torch.stack (diceList,0).mean()
                print (" mean dice score {:3f}  ".format(meandice ))
                
    def train_epochs(self,trainLoader,valLoader):
        for epoch in range (1, self.cfg.train.epochs+1):
            self.train_epoch(dataLoader=trainLoader,epoch=epoch,model=self.model)
            self.eval(dataLoader=valLoader,model=self.model,epoch=epoch)
            # self.scheduler.step()
    def loadModel(self):
        if self.cfg.model=="unet":
            return smp.Unet(
            encoder_name="se_resnext50_32x4d", 
            encoder_weights="imagenet", 
            classes=1, 
            activation=None,
        )
    def loadOptimizer(self,model):
        if self.cfg.train.optimizer.name=="Adam":
            op=optim.Adam(model.parameters(),lr=self.cfg.train.optimizer.lr, weight_decay=self.cfg.train.optimizer.weight_decay)
            scheduler = StepLR(op, step_size=1, gamma=0.1,verbose=True)
            return op,scheduler
    def loadCriterion(self):
        return modelUtils.MixedLoss(gamma= 2.0,alpha=1)
       
         
    


            
            




    
