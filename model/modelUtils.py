import torchmetrics
from torchmetrics import Dice, JaccardIndex
import numpy as np
class Metric():
    def __init__(self,cfg,device) -> None:
        self.cfg=cfg
        # self.metrics=[]
        self.device=device
        self.iou_fn=torchmetrics.JaccardIndex(num_classes=2, task="binary", average="macro").to(self.device)
        self.dice_fn=torchmetrics.Dice(num_classes=2, average="macro").to(self.device)
    def computeMetric(self,output,target):
        metric={}
        metric["iou"]=self.iou_fn(output,target)
        metric["dice"]=self.iou_fn(output,target)
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


