
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import tqdm
import sys
sys.path.append("../data")
from data import dataUtil,dataLoader
from . import modelUtils,models,metric
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
        self.save_plot=modelUtils.Save_plot(cfg=cfg)
        self.save_best_model=modelUtils.SaveBestModel(cfg=cfg)
        self.train_loader,self.val_loader=dataLoader.loadData(cfg=self.cfg)
        self.tta_test_loader=dataLoader.tta_testLoader(cfg=cfg)
        self.test_size=min(self.cfg.sampler.randomSampler.number,len(self.tta_test_loader.dataset))
    def train_epoch(self,dataLoader,model,epoch):
        losses=metric.AverageMeter()
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
                losses.update(loss.item())
                loss.backward()   
                self.optimizer.step()
                if idx % log_interval == 0:                   
                        tepoch.set_postfix(loss=loss.item())
        return {
            "loss":losses.avg
        }
        print ("-"*100)

    def eval (self,dataLoader,model,epoch):
        model.eval()
        losses=metric.AverageMeter()       
        diceMetric=metric.Metric()
        diceList=[]
        with torch.no_grad():
            with tqdm(dataLoader, unit="batch") as tepoch:
                for idx, (_,x,y_true) in enumerate(tepoch):
                    x=x.to(self.device).float()
                    y_hat=model(x).squeeze(1)     
                    y_true=y_true.to(self.device).float()
                    y_hat=torch.sigmoid(y_hat)
                    dice,term,denom=diceMetric.compDice(y_hat,y_true)  
                    loss=self.criterion(y_hat,y_true)
                    losses.update(loss.item())
                    diceList.append(dice)      
                    if idx%4==0:
                        tepoch.set_postfix(curdice=dice.item(),term=term.item(),denom=denom.item())
                meandice=torch.stack (diceList,0).mean().item() 
                print (" mean dice score {:3f}  ".format(meandice ))
                return {
                    "loss":losses.avg,
                    "dice":meandice

                }
    def tta_eval(self,dataLoader,model,epoch):
        print ("Eval tta epoch {}".format(epoch))
        model.eval()
        out_losses=[]
        diceMetric=metric.Metric()
        init=True
        out_preds=torch.zeros(self.test_size,self.cfg.image.size,self.cfg.image.size).to(self.device)
        out_true=torch.zeros(self.test_size,self.cfg.image.size,self.cfg.image.size).to(self.device)
        with torch.no_grad():
            for i in range (self.cfg.tta.times):                             
                losses=[]
                predmask=[]
                truemask=[]
                with tqdm(dataLoader,unit="batch") as tepoch:
                    for idx, (_,x,y_true) in enumerate (tepoch):
                        x=x.to(self.device).float()
                        y_hat=model(x).squeeze(1)     
                        y_true=y_true.to(self.device).float()
                        y_hat=torch.sigmoid(y_hat)
                        loss=self.criterion(y_hat,y_true)
                        losses.append (loss.item())
                        predmask.append(y_hat)                   
                        truemask.append(y_true)                        
                        if idx%4==0:
                            tepoch.set_postfix(loss = loss.item())
                predmask=torch.concat(predmask,dim=0)  
                truemask=torch.concat(truemask,dim=0)
                losses=sum(losses)/len(losses)              
                out_preds+=predmask
                out_true+=truemask
                out_losses.append(losses)
            out_preds=(out_preds/self.cfg.tta.times).sigmoid()
            out_true=out_true/self.cfg.tta.times
            out_losses=sum(out_losses)/len(out_losses)
            dice,_,_=diceMetric.compDice(out_preds,out_true)  
            print (" Avg dice with tta :{}".format(dice))
            return {
                "dice":dice.item(),
                "loss":[]
                # "loss":out_losses
                }
    def train_epochs(self):
        train_metrics=[]
        val_metrics=[]
        for epoch in range (1, self.cfg.train.epochs+1):
            train_metric=self.train_epoch(dataLoader=self.train_loader,epoch=epoch,model=self.model)
            if self.cfg.tta.usetta=="False":
                val_metric=self.eval(dataLoader=self.val_loader,model=self.model,epoch=epoch)
            else:
                val_metric=self.tta_eval(dataLoader=self.tta_test_loader,model=self.model,epoch=epoch)
            train_metrics.append(train_metric)
            val_metrics.append(val_metric)
            modelUtils.recordTraining(epoch=epoch,cfg=self.cfg, metric=val_metric)
            self.save_best_model(val_metric,epoch,self.model,self.optimizer)

        self.save_plot.save(train_metrics=train_metrics,val_metrics=val_metrics)      
            # self.scheduler.step()
    def loadModel(self):
        if self.cfg.model=="unet":
            return smp.Unet(
            encoder_name="se_resnext50_32x4d", 
            encoder_weights="imagenet", 
            classes=1, 
            activation=None,
        )
        elif self.cfg.model=="denseUnet":
            return models.DenseUNet(imageSize=self.cfg.image.size,numclass=1)
    def load_model_ckp(self):
        state_dict=torch.load(os.path.dirname(os.path.abspath(__name__))+"/model/output/best_model.pth")
        print ("Load model check point with dice score = {}".format(state_dict["metric"]["dice"]))
        return state_dict["model_state_dict"]
    def loadOptimizer(self,model):
        if self.cfg.train.optimizer.name=="Adam":
            op=optim.Adam(model.parameters(),lr=self.cfg.train.optimizer.lr, weight_decay=self.cfg.train.optimizer.weight_decay)
            scheduler = StepLR(op, step_size=1, gamma=0.1,verbose=True)
            return op,scheduler
    def loadCriterion(self):
        return metric.MixedLoss(gamma= 2.0,alpha=1)
       
    def visualizeNoLabel(self,input):
        self.model.eval()
        with torch.no_grad():
            input=input.to(self.device)
            y_pred=self.model(input)
            y_pred[y_pred>=0.5]=1.
            y_pred[y_pred<0.5]=0.   
            if y_pred.max()>0:
                fig,ax=plt.subplots(3)
                plt.figure(figsize=(16, 8))             
                ax[0].imshow(np.array(input.transpose(1,2,0)))
                ax[0].set_title("Original image")
                ax[1].imshow(np.array(y_pred.squeeze(0) ))
                ax[1].set_title("detected disease")
                ax[2].imshow(np.array(input.transpose(1,2,0)))
                ax[2].imshow(np.array(mask.squeeze(0)*100 ), alpha=0.25)
                ax[2].set_title(" Combine")
                fig.tight_layout()
            else:
                
                plt.figure(figsize=(16, 8))             
                plt.imshow(np.array(input.transpose(1,2,0)))
                plt.title("detecting no disease")
            plt.show()
            fig.savefig(os.path.dirname(os.path.abspath(__name__))+"/model/output/unlabel_predict.png")
    def visualizeWithLabel(self,input,target):
        self.model.eval()
        with torch.no_grad():
            input=input.to(self.device)
            y_pred=self.model(input)
            y_pred[y_pred>=0.5]=1.
            y_pred[y_pred<0.5]=0.  
            fig,ax=plt.subplots(3)
            ax[0].imshow(np.array(input.transpose(1,2,0)))
            ax[1].imshow(np.array(y_pred.squeeze(0) ))
            ax[1].set_title("Predicting")
            ax[2].imshow(np.array(target))
            ax[2].set_title(" Ground truth")
            fig.tight_layout()
            plt.show()
            fig.savefig(os.path.dirname(os.path.abspath(__name__))+"/model/output/labeled_predict.png")




    


            
            




    
