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
# load record.json file
path= os.path.dirname(os.path.abspath(__name__)) +"/output/record.json"
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
        
        self.best_dice_score = int (record.best_dice_score)
        self.cfg=cfg
        
        
    def __call__(
        self, metric, 
        epoch, model, optimizer
    ):
        if int(self.cfg.sampler.randomSampler)>=100000 and metric.dice>self.best_dice_score:
            write_json(key=best_dice_score,val=metric.dice,filename=path)
            self.best_dice_score=metric.dice
            now = datetime.now() 
            dt_string = now.strftime("%d-%m-%Y-%H:%M:%S")
            file_path=os.path.dirname(os.path.abspath(__name__))+"/model/output/best_model.pth")    
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
        if int(self.cfg.sampler.randomSampler)>=100000 :
            now = datetime.now() 
            dt_string = now.strftime("%d-%m-%Y-%H:%M:%S")
            file_path=os.path.dirname(os.path.abspath(__name__))+"/model/output/learning_analysis/{}.png".format(dt_string)
            train_losses=[]
            val_losses=[]
            val_dice=[]
            for metric in train_metric:
                train_losses.append(metric["loss"])
            for metric in val_metric:
                val_losses.append(metric["loss"])
                val_dice.append(metric["dice"]*100)
            fig,ax=plt.subplots(2)
            ax[0].plot(train_losses,label="train loss")
            ax[0].plot(val_losses,label="val loss")
            ax[0].set(xlabel="epoch")
            ax[0].set_title("Loss statistic")
            ax[1].plot(val_dice,label="val dice score")
            ax[1].set_title("Dice statistic")
            ax[1].set(xlabel="epoch",ylabel="percent")
            fig.tight_layout()
            plt.show()
            fig.savefig(file_path)


def write_json(key,val, filename):
    
    with open(filename,'r+') as file:
          # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data[key]=val
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 4)


         

        

            

