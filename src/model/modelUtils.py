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



def save_metrics_and_models(metrics,models):      
    multiple_train_metrics=metrics["train_stats"]
    multiple_val_metrics=metrics["val_stats"]
    
    now = datetime.now() 
    folder=os.path.dirname(os.path.abspath(__name__))+"/output/"
    models_folder=os.path.join(folder,"models") 
    os.makedirs( models_folder ,exist_ok=True)
    # ../chexpert_classification/chexpert/output/models
    dice_info_path=os.path.join(folder,"k_fold_dice.txt")
    plots_folder=os.path.join(folder,"plot") 
    os.makedirs( plots_folder ,exist_ok=True)

    with open(dice_info_path, 'w') as file:
    # Write text to the new file
        file.write('This is evaluation dice score for each fold.\n')

    for i in range(len(models)):
        mean_dice_of_epochs=np.mean([data["dice"] for data in multiple_val_metrics[i]])
        highest_mean_dice=np.max([data["dice"] for data in multiple_val_metrics[i]] )
        torch.save({
                    'fold':i+1,
                    'train_metric':multiple_train_metrics[i],
                    'val_metric':multiple_val_metrics[i],

                    'model_state_dict': models[i],
                  
                    }, os.path.join(models_folder,f"fold_{i+1}.pth"))
        with open(dice_info_path, 'a') as file:
            file.write(f'Fold {i+1} mean dice of epochs {mean_dice_of_epochs}, highest dice {highest_mean_dice}.\n')


    


         

        

            

