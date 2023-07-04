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
class SaveBestModel:
    # this class only work for each training
    """
    Class to save the best model while training. If the current epoch's 
    validation dice is higher than saved metric, then save the
    model state.
    """
    def __init__(
        self, cfg,best_valid_dice=-float('inf'),threshold=0.75
    ):
        self.best_valid_dice = best_valid_dice
        self.cfg=cfg
        self.threshold=threshold
        
    def __call__(
        self, metric, 
        epoch, model, optimizer, criterion
    ):
        if int(self.cfg.sampler.randomSampler)>=100000 and metric.dice>self.threshold:
            now = datetime.now() 
            dt_string = now.strftime("%d-%m-%Y-%H:%M:%S")
            file_path=os.path.dirname(os.path.abspath(__name__))+"/model/output/model_state_dict/dice:{}-{}.pth".format(metric["dice"],dt_string)
            with open(file_path, 'w') as file:
                file.write()
            if metric["dice"] > self.best_valid_dice:
                self.best_valid_dice= metrics["dice"]
                print(f"\nBest validation dice score: {self.best_valid_dice}")
                print(f"\nSaving best model for epoch: {epoch}\n")
                torch.save({
                    'metric':metric,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    
                    }, file_path)
class save_plot():
    def __init__(self,cfg,threshold):
        self.cfg=cfg
        self.threshold=threshold
    def save(train_metrics,val_metrics):
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


            

        

            

