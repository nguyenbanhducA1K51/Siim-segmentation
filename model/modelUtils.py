import torchmetrics
from torchmetrics import Dice, JaccardIndex
import numpy as np
import torch.nn as nn
import torch as t
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

class CombineLoss(nn.Module):
    def __init__(self,gamma) -> None:
        super(CombineLoss,self).__init__()
        self.gamma=gamma

    def forward(self, output,target):
        pass
    def diceLoss(self,output,target):
        # output is in range (0,1)
        output=torch
        return output*target
    def focalLoss(self,output,target):
        # output is in range (0,1)
        ep=1e-4
        output=output
        if output<0.5:
            output+=1e-10
        else:
            output-=1e-10
        loss=-t.pow(1-output,self.gamma)*target*t.log(output)-t.pow(output,self.gamma)*(1-target)* t.log(1-output)
        print ("loss {} ".format(loss))
        return loss
        





# output=torch.clamp(output,min=1e-10, max=0.9999999)

#         positive=torch.sum(target,dim=0)
#         negative= target.size()[0]-positive
#         positive_factor= (1-self.beta)/ (1-self.beta**positive+1e-10)

#         negative_factor=(1-self.beta)/ (1-self.beta**negative+1e-10)

#         positive_factor=  positive_factor.unsqueeze(0).to(self.device)
#         negative_factor= negative_factor.unsqueeze(0).to(self.device)

#         positive_factor=torch.repeat_interleave(positive_factor, torch.tensor([target.size()[0]]).to(self.device), dim=0)
#         negative_factor=torch.repeat_interleave(negative_factor, torch.tensor([target.size()[0]]).to(self.device), dim=0)

#         # print ("positive_factor (1-beta)/ (1-beta**positive+1e-10) \n{}".format(positive_factor))
#         # print ("negative_factor (1-beta)/ (1-beta**negative+1e-10)\n{}".format(negative_factor))
#         loss=-positive_factor*target*torch.log(output)-(1-target)* negative_factor*torch.log(1-output)







