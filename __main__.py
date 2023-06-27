# from ..data import dataUtil
# print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
import sys
import torch
import json,os
from  model import siim
from  data import dataLoader
from torch.nn import functional as F
from easydict import EasyDict as edict
cfg_path=os.path.dirname(os.path.abspath(__name__))+"/config/config.json"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device="cpu"
with open(cfg_path) as f:
    cfg = edict(json.load(f))
trainLoader,valLoader=dataLoader.loadData(cfg=cfg)
model=siim.SIIMnet(cfg=cfg,device=device)
model.train_epochs(trainLoader=trainLoader,valLoader=valLoader)
