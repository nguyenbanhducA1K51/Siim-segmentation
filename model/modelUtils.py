import torchmetrics
from torchmetrics import Dice, JaccardIndex
from torchmetrics.functional import dice as d
import numpy as np
import torch.nn as nn
import torch    
import torch.nn.functional as F 
from datetime import datetime
import matplotlib.pyplot as plt
import os 
from easydict import EasyDict as edict
import json
# load record.json file
path= os.path.dirname(os.path.abspath(__name__)) +"/model/output/record.json"
with open(path) as f:
    record = edict(json.load(f))
class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation dice is higher than saved metric, then save the
    model state.
    """
    def __init__(
        self, cfg
    ):
        
        self.best_dice_score = record.best_dice_score
        self.cfg=cfg
        
        
    def __call__(
        self, metric, 
        epoch, model, optimizer
    ):
        if metric["dice"]>self.best_dice_score:
            self.best_dice_score=metric["dice"]
            write_json(key="best_dice_score",val=metric["dice"],filename=path)       
            now = datetime.now() 
            dt_string = now.strftime("%d-%m-%Y-%H:%M:%S")
            file_path=os.path.dirname(os.path.abspath(__name__))+"/model/output/best_model.pth"    
            print(f"\nBest validation dice score: {self.best_dice_score}")
            print(f"\nSaving best model for epoch: {epoch}\n")
            torch.save({
                "times":dt_string,
                'metric':metric,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                
                }, file_path)
class Save_plot():
    def __init__(self,cfg,threshold=0.5):
        self.cfg=cfg
        self.threshold=threshold
    def save(self,train_metrics,val_metrics):
        # if self.cfg.sampler.randomSampler.number>=100000 :
            now = datetime.now() 
            dt_string = now.strftime("%d-%m-%Y-%H:%M:%S")
            file_path=os.path.dirname(os.path.abspath(__name__))+"/model/output/learning_analysis/{}.png".format(dt_string)
            train_losses=[]
            val_losses=[]
            val_dice=[]
            for metric in train_metrics:
                train_losses.append(metric["loss"])
            for metric in val_metrics:
                val_losses.append(metric["loss"])
                val_dice.append(metric["dice"]*100)
            fig,ax=plt.subplots(2)
            ax[0].plot(train_losses,label="train loss",color="green",marker="o")
            ax[0].plot(val_losses,label="val loss",color="blue",marker="o")
            ax[0].set(xlabel="epoch")
            ax[0].set_title("Loss statistic")
            ax[1].plot(val_dice,label="val dice score",color="blue",marker="o")
            ax[1].set_title("Dice statistic")
            ax[1].set(xlabel="epoch",ylabel="percent")
            ax[0].legend()
            ax[1].legend()
            fig.tight_layout()
            plt.show()
            fig.savefig(file_path)


def write_json(key,val, filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    data[key] = val # 

    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def recordTraining(epoch=0,cfg=None, metric=None):  
    
    if int(cfg.sampler.randomSampler.number)>=100000:
        now = datetime.now()   
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        filePath=os.path.dirname(os.path.abspath(__name__))+"/model/output/recordTraining.csv"
        with open(filePath, "a") as file:
            
            mean_dice=round(metric["dice"] ,3)
            model=cfg.model  
            sample=cfg.sampler.randomSampler.number  
            epoch=str(epoch)+"/"+str(cfg.train.epochs)
            op=cfg.train.optimizer.lr
            lr=cfg.train.optimizer.lr
            criterion=cfg.criterion                   
            batch_size=cfg.train.batch_size
                      
            use_progressive_training=cfg.progressive_train.use
            totalProgressiveEpoch=cfg.progressive_train.epochs
            progressiveOP=cfg.progressive_train.optimizer.name
            progressivelr=cfg.progressive_train.optimizer.lr        
            usetta=cfg.tta.usetta 
            tta_times=cfg.tta.times 

            finalString=""
            finalString+=dt_string+","
            finalString+=str(mean_dice)+","
            finalString+=str(model)+","
            finalString+=str(sample)+","
            finalString+=str(epoch)+","
            finalString+=str(op)+","
            finalString+=str(lr)+","
            finalString+=str(criterion)+","
            finalString+=str(batch_size)+","
            
            finalString+=str(use_progressive_training)+","
            finalString +=str(totalProgressiveEpoch)+","
            finalString+=str(progressiveOP)+","
            finalString+=str(progressivelr)+","
            finalString+=str(usetta)+","   
            finalString+=str(tta_times)+","    
            # print (finalString)
            file.write('\n'+finalString)


    


         

        

            

