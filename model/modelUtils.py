import torchmetrics
from torchmetrics import Dice, JaccardIndex
from torchmetrics.functional import dice as d
import numpy as np
import torch
class Metric():
    def __init__(self,cfg,device) -> None:
        self.cfg=cfg
        # self.metrics=[]
        self.device=device
        self.iou_fn=torchmetrics.JaccardIndex(num_classes=2, task="binary", average="macro").to(self.device)
    def computeMetric(self,output,target):
        metric={}
        output=output.sigmoid()
        output[output>=0.5]=1
        output[output<0.5]=0
    
        print ( "sum target {}, sum output{} sum product {}".format (target.sum(),output.sum(), (target*output).sum()))
        if output.sum()==0. and target.sum() ==0:          
            metric["dice"]=1
        else:
            dice=2*(target*output).sum()/ (target.sum()+output.sum() )
            metric["dice"]=dice.item()
            # d(output,target,average="micro").item()
        metric["iou"]=self.iou_fn(output,target).item()
    
        return metric
    
class AverageMeter():
    def __init__(self) -> None:
        self.reset()
    def reset(self):
        self.cur=0
        self.ls=[]
        self.avg=0
    def update(self,item):
        self.cur=item
        self.ls.append(item)
        self.avg=np.mean(self.ls)


