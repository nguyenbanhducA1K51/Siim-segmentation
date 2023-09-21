
import numpy as np
import cv2
import torch
import tqdm
import sys
import copy
import glob
import random

import os


from torch.optim import Adam
import torch.optim as optim
import segmentation_models_pytorch as smp
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR,StepLR,CosineAnnealingLR

sys.path.append("../data")
sys.path.append("../")
sys.path.append("/root/repo/Siim-segmentation/src")

from dataLoader import SIIMDataset,UpSampler
from models.unet import ResnetSuperVision
from models.resunet import ResUnet
import metric
import dataLoader
from losses.combine import MultipleLoss

class SIIMnet():
    def __init__(self,cfg,device,fold) :
        self.fold=fold
        self.cfg=cfg
        self.device=device
        self.model=self.loadModel().to(self.device)
        self.optimizer,self.scheduler=self.loadOptimizer(model=self.model)
        self.criterion=self.loadCriterion()
        self.dice=metric.DiceMetric()
    def train_epoch(self,dataLoader,model,epoch):
        losses=metric.AverageMeter()
        # diceMetric=metric.Metric()
        
        diceList=[]
        log_interval=self.cfg.train.log_interval
        model.train()
        model.to("cuda")
        with tqdm(dataLoader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            for idx,(x,y_true) in enumerate(tepoch) :
                self.optimizer.zero_grad()
                x=x.to(self.device).float()

                y_true=y_true.unsqueeze(1).to(self.device).float()  

                c=y_true.clone().cuda()
                # is_empty = torch.tensor([y_true.sum(1) > 0], dtype=torch.float32, device='cuda')
                is_empty=y_true.view( y_true.shape[0],-1).sum(1)
                is_empty=(is_empty>0).float().cuda()


                y_hat, pred_empty=model(x)
                pred_empty=pred_empty.squeeze(1)

                loss=self.criterion(y_hat,y_true,pred_empty,is_empty)
                losses.update(loss.item())
                loss.backward()  
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) 
                
                dice=self.dice(y_hat,y_true) 
                diceList.append(dice.item())  
                self.optimizer.step()
                if idx % log_interval == 0:                   
                        tepoch.set_postfix(loss=loss.item(),dice=dice.item(),grad=grad_norm.item())
        # meandice=torch.stack (diceList,0).mean().item() 
        meandice=np.mean(diceList)
        print (f"Train mean dice {meandice}")
        return {
            "loss":losses.avg,
            "dice":meandice
        }
    

    def eval (self,dataLoader,model,epoch):
        print (f"EVAL on epoch {epoch}")
        model.eval()
        model.to("cuda")
        losses=metric.AverageMeter()       
        diceList=[]
        with torch.no_grad():
            with tqdm(dataLoader, unit="batch") as tepoch:
                for idx, (x,y_true) in enumerate(tepoch):
                    x=x.to(self.device).float()
                    y_true=y_true.unsqueeze(1).to(self.device).float()  
                    is_empty=y_true.view( y_true.shape[0],-1).sum(1)
                    is_empty=(is_empty>0).float().cuda()
                    y_hat, pred_empty=model(x)
                    pred_empty=pred_empty.squeeze(1)
                    loss=self.criterion(y_hat,y_true,pred_empty,is_empty)            

                    losses.update(loss.item())
                    dice=self.dice(y_hat,y_true) 
                    diceList.append(dice.item())      
                    if idx%4==0:
                        tepoch.set_postfix(curdice=dice.item())

                meandice=np.mean(diceList)
                print (" mean dice score {:3f}  ".format(meandice ))
                return {
                    "loss":losses.avg,
                    "dice":meandice

                }
    def train(self):

            trainset=SIIMDataset(cfg=self.cfg,mode="train",fold=self.fold)
            sampler= UpSampler (trainset,self.cfg.train.epochs)

            valset=SIIMDataset(cfg=self.cfg,mode="val",fold=self.fold)
            print (f"len trainset{len(trainset)}, len val {len(valset)}")
            train_loader = torch.utils.data.DataLoader(
                        trainset, 
                        batch_size=self.cfg.train.batch_size,sampler=sampler)
            val_loader = torch.utils.data.DataLoader(
                        valset,
                        batch_size=self.cfg.train.batch_size )
                    
            train_metrics,val_metrics=self.train_epochs(train_loader,val_loader,self.fold)
        

            mean_dices_of_epochs=np.mean([data["dice"] for data in val_metrics]),
            highest_mean_dice=np.max([data["dice"] for data in val_metrics] ),
            print (f"Finish train on fold {self.fold} with mean dice of epochs {mean_dices_of_epochs}, highes mean auc {highest_mean_dice}")
            # modelUtils.save_metrics_and_models({"train_stats":train_metrics,"val_stats":val_metrics},self.model.state_dict())



    def train_epochs(self,train_loader,val_loader,fold):
        print (f"Training on FOLD{fold}")
        train_metrics=[]
        val_metrics=[]

        # for early stopping
        best_val_dice=float('-inf')
        counter=0
        for epoch in range (0, self.cfg.train.epochs):
            train_loader.sampler.set_epoch(epoch)
            print (f"epoch{epoch+1}of {self.cfg.train.epochs}")
            train_metric=self.train_epoch(dataLoader=train_loader,epoch=epoch,model=self.model)
            val_metric=self.eval(dataLoader=val_loader,model=self.model,epoch=epoch)         
            train_metrics.append(train_metric)
            val_metrics.append(val_metric)

            
            best_val_dice=max(val_metric["dice"],best_val_dice)
            if val_metric["dice"]< best_val_dice:
                counter+=1
            else:
                counter=0
            if counter >self.cfg.train.early_stop.patient:
                print (f"early stop on epoch {epoch}")
                break
            self.scheduler.step()
        return train_metrics, val_metrics
           
    def loadModel(self):
        if self.cfg.model=="unet":
            # return ResnetSuperVision(1, backbone_arch='resnet34')
            return ResUnet(1)
        
   
    
    def loadOptimizer(self,model):
        if self.cfg.train.optimizer.name=="Adam":
            op=optim.Adam(model.parameters(),lr=self.cfg.train.optimizer.lr, weight_decay=self.cfg.train.optimizer.weight_decay)
            scheduler = CosineAnnealingLR(op, T_max=8, eta_min=0.000001)

            return op,scheduler
    def loadCriterion(self):
       
        return MultipleLoss()
       
    
    def eval_and_visualize_testset_with_label(self):
        # here we use test set with label to estimate models performance on real test case
        print ("EVALUATING k-folds on test set")      
        num_visualize=10       
        samples=[]

        data_loader=torch.utils.data.DataLoader(
                    self.testset, 
                    batch_size=self.cfg.train.batch_size)
        
        models_globs=f"{os.environ['proj_path']}/output/models/*.pth"
        model_paths=glob.glob(models_globs)
        assert (len(model_paths)>0),"empty model path"
        print (f"Total models {len(model_paths)}")
        models=[]
        for path in model_paths:
            statedict=torch.load(path)
            model=copy.deepcopy(self.model)
            model.load_state_dict(statedict["model_state_dict"])
            model.eval()
            models.append (model)
     
        diceMetric=metric.Metric()
        diceList=[]
        with torch.no_grad():
            with tqdm(data_loader, unit="batch") as tepoch:
                for idx, (x,y_true) in enumerate(tepoch):
                    x=x.to(self.device).float()
                    y=[]
                    for model in models:
                        y_hat=model(x).squeeze(1)  
                        y.append(y_hat)
                    y_pred=torch.mean(torch.stack(y), dim=0)
                    y_true=y_true.to(self.device).float()
                    y_pred=torch.sigmoid(y_pred)

                    random_number = random.random()
                    if random_number>0.5 and len(samples)<num_visualize:
                        samples.append((x,y_pred,y_true) )
                  
                    dice,term,denom=diceMetric.compDice(y_pred,y_true)     
                    diceList.append(dice)      
                    if idx%4==0:
                        tepoch.set_postfix(curdice=dice.item(),term=term.item(),denom=denom.item())
                meandice=torch.stack (diceList,0).mean().item() 
                print (f" Average dice when evluate on test set : {meandice}")

            # Visualize

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            # print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()



    


            
            




    
