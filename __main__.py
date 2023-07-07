
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import sys
import torch
import json,os
from  model import siim
from  data import dataLoader
from torch.nn import functional as F
from easydict import EasyDict as edict
import warnings
import logging

warnings.filterwarnings("ignore")
cfg_path=os.path.dirname(os.path.abspath(__name__))+"/config/config.json"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device="cpu"
record_path=os.path.dirname(os.path.abspath(__name__))+"/model/output/best_model.pth"
print (record_path)
state_dict=torch.load(os.path.dirname(os.path.abspath(__name__))+"/model/output/best_model.pth")
print ("record model check point with dice score = {}".format(state_dict["metric"]["dice"]))

with open(cfg_path) as f:
    cfg = edict(json.load(f))

model=siim.SIIMnet(cfg=cfg,device=device)
model.train_epochs()