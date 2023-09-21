
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import sys
import torch
import json,os
from torch.nn import functional as F
from easydict import EasyDict as edict
import warnings
import logging
import argparse

sys.path.append("..")
import siim
from  src import dataLoader


warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--fold', type=str, default=1)
    parser.add_argument('--mode',type=str,default="train")
    return parser.parse_args()

if __name__=="__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parse_args()

    with open(args.config) as f:
        cfg = edict(json.load(f))       
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=siim.SIIMnet(cfg=cfg,fold=args.fold,device=device)
    if args.mode=="train":
        model.train()
    else:
        model.test()