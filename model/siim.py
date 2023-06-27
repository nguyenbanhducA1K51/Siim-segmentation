# print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import tqdm
import sys
sys.path.append("../data")
sys.path.append("../model")
from data import dataUtil
from model import modelUtils, models
from torch.optim import Adam
import torch.optim as optim
import segmentation_models_pytorch as smp

class SIIMnet():
    def __init__(self,cfg,device) :
        self.cfg=cfg
        self.device=device
        self.metric=modelUtils.Metric(cfg=cfg,device=self.device)
        self.model=self.loadModel().to(self.device)
        self.optimizer=self.loadOptimizer(model=self.model)
        self.criterion=self.loadCriterion()
        
    def train_epoch(self,dataLoader,model,epoch):
        log_interval=self.cfg.train.log_interval
        model.train()
        for idx, (oriIdx,x,y_true) in enumerate(dataLoader):
            x=x.to(self.device).float()
            y_hat=model(x).squeeze(1)
            y_true=y_true.to(self.device).float()  
            self.optimizer.zero_grad()
            loss=self.criterion(y_hat,y_true)
            loss.backward()
            self.optimizer.step()
            if idx % log_interval == 0:
                    print(' index {} Train Epoch: {} [{}/{} ({:.0f}%)]\t  Loss {: .5f}'.format( idx,
                        epoch, idx * len(x), len(dataLoader.dataset), 
                        100. * idx / len(dataLoader),loss.item() ))
        
    def eval (self,dataLoader,model,epoch):
        model.eval()
        iouAverage= modelUtils.AverageMeter()
        diceAverage=modelUtils.AverageMeter()
        lossAverage=modelUtils.AverageMeter()
        with torch.no_grad():
            for idx, (_,x,y_true) in enumerate(dataLoader):
                x=x.to(self.device).float()
                y_hat=model(x).squeeze(1)
                
                
                y_true=y_true.to(self.device)

                loss=self.criterion(y_hat,y_true)
                print (y_hat)
            
                metric=self.metric.computeMetric(output=y_hat,target=y_true)
                lossAverage.update(loss.item())
                diceAverage.update(metric["dice"])
                iouAverage.update(metric["iou"])
            print ("Mean loss {:3f}, Mean IoU : {:3f}, Mean Dice :{:3f} ".format(lossAverage.avg,iouAverage.avg,diceAverage.avg) )
            return lossAverage.avg,iouAverage.avg,diceAverage.avg
    def train_epochs(self,trainLoader,valLoader):
        for epoch in range (1, self.cfg.train.epochs+1):

            self.train_epoch(dataLoader=trainLoader,epoch=epoch,model=self.model)
            self.eval(dataLoader=valLoader,model=self.model,epoch=epoch)
    def loadModel(self):
        if self.cfg.model=="unet":
            # return models.Unet(n_classes=2)
            return smp.Unet(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES), 
            activation=ACTIVATION,
        )
    def loadOptimizer(self,cfg,model):
        if cfg.train.optimizer=="Adam":
            return optim.Adam(model.parameters(),lr=self.cfg.train.optimizer.lr, weight_decay=self.cfg.train.optimizer.weight_decay)
    def loadCriterion(self):
        if self.cfg.criterion=="bce":
            return nn.BCEWithLogitsLoss()
         
    


            
            




    
