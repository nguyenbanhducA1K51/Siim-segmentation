
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


# Suppress all warnings
warnings.filterwarnings("ignore")
cfg_path=os.path.dirname(os.path.abspath(__name__))+"/config/config.json"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device="cpu"
with open(cfg_path) as f:
    cfg = edict(json.load(f))
trainLoader,valLoader=dataLoader.loadData(cfg=cfg)

