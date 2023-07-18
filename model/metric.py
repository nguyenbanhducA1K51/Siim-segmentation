import torchmetrics
from torchmetrics import Dice, JaccardIndex
from torchmetrics.functional import dice as d
import numpy as np
import torch.nn as nn
import torch    
import torch.nn.functional as F 
class focalLoss(nn.Module):
    def __init__(self,gamma=3,alpha=1):
        super().__init__()
        self.gamma=gamma
        self.alpha=alpha
        
    def forward(self,input,target):
       # -log (p).t -log(1-p)*(1-t)
        ep=1e-7
        clampinput=torch.clamp(input,min=ep, max=1-ep) 
        coef1= self.alpha * ((1 - clampinput) ** self.gamma)
        coef2=  self.alpha * ( clampinput ** self.gamma )
        loss = - coef1* torch.log(clampinput) * target  - coef2 * torch.log(1-clampinput) * (1-target)
        
        return loss.mean()


class weightBinaryloss(nn.Module):
    def __init__(self,beta=3):
        super().__init__()
        self.beta=beta
    def forward(self,input,target):
        ep=1e-7
        clampinput=torch.clamp(input,min=ep, max=1-ep)  
        loss=-self.beta*target*torch.log(clampinput)-(1-target)* torch.log(1-clampinput)
        return loss.mean()
   
def dice_coef(input, target):
    # input in range(0,1)
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    dice=(2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)   
    return dice
class MixedLoss(nn.Module):
    def __init__(self,  gamma=3,alpha=1):
        super().__init__()
        self.weightbce=weightBinaryloss()
        self.focal = focalLoss(alpha,gamma)
        self.bce=nn.BCELoss(reduction="mean")
        

    def forward(self, input, target):
        dice= 1-dice_coef(input, target)
        bce=self.bce(input,target)
        weightbce=self.weightbce(input,target)
        focal=self.focal(input, target)
        loss=bce+dice+focal

        return loss.mean()
          
class Metric():
    def __init__(self):
        self.positivedice=0
        self.dicelist=[]
        self.meandice=0
    def compDice(self,y_pred,y_target):
        batch_size=y_target.shape[0]
        y_pred=y_pred.view(batch_size,-1)
        y_pred[y_pred>=0.5]=1.
        y_pred[y_pred<0.5]=0.
        y_target=y_target.view(batch_size,-1)
       
        term=2* (y_target*y_pred).sum(-1)+1e-4
        denom=y_target.sum(-1)+y_pred.sum(-1)+1e-4
        dice=term/denom
        return dice.mean(),term,denom
        
    def getMeanDice(self):
            
            return torch.mean(torch.stack(self.dicelist)) 

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
