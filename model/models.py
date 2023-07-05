
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import sys
from collections import OrderedDict

class DenseUNet (nn.Module):
    def __init__ (self,imageSize,numclass):
        super(DenseUNet, self).__init__()
        assert imageSize%32==0 ,"input image must be divisible by 32"
        self.numclass=numclass
        self.imageSize=imageSize
        denseNet=models.densenet121(weights="DenseNet121_Weights.IMAGENET1K_V1" )
        self.sequenceLayer=nn.Sequential( OrderedDict ([ 
                ("conv0", nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) ),
                ( "norm0", nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) ),
                ("relu0", nn.ReLU(inplace=True) ),
                ( "pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False) ) ]) )
        self.denseblock1=denseNet.features.denseblock1
        self.transition1=denseNet.features.transition1
        self.denseblock2=denseNet.features.denseblock2
        self.transition2=denseNet.features.transition2
        self.denseblock3=denseNet.features.denseblock3
        self.transition3=denseNet.features.transition3
        self.denseblock4=denseNet.features.denseblock4
        self.norm5=denseNet.features.norm5
        self.pool=nn.MaxPool2d(2)
        self.bottleNeck=DoubleConv(1024,1024,1024)       
        self.up1= Up(1024,256)
        self.up2=Up(256,128)
        self.up3=Up(128,64)
        self.decodeconv1=nn.Conv2d(256*2,256,kernel_size=3,padding="same")
        self.decodeconv2=nn.Conv2d(128*2,128,kernel_size=3,padding="same")
        self.decodeconv3=nn.Conv2d(64*2,64,kernel_size=3,padding="same")
        self.almostlast=nn.Sequential( OrderedDict ([ 
            ("upconv1", nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)),
            ("conv",nn.Conv2d(64,self.numclass,kernel_size=3,padding="same")),
        ("norm",nn.BatchNorm2d(self.numclass))
        ]))              
        self.sigmoid=nn.Sigmoid()            
    def forward(self,x):
        #encoder
        respath1=self.sequenceLayer(x)
        #64*128*128
        x=self.denseblock1(respath1)
        #256, 128, 128
        respath2=self.transition1(x)
        #128, 64, 64
        x=self.denseblock2(respath2)
        #512, 64, 64
        respath3=self.transition2(x)
        #256, 32, 32]
        x=self.denseblock3(respath3)
        # 1024, 32, 32
        respath4=self.transition3(x)
        #512,16,16
        x=self.denseblock4(respath4)
        #1024,16,16
        x=self.norm5(x)    
        x= self.up1(x)
        x=torch.cat([respath3,x],dim=1)
        x=self.decodeconv1(x)
        x= self.up2(x)
        x=torch.cat([respath2,x],dim=1)
        x=self.decodeconv2(x)
        x=self.up3(x)
        x=torch.cat([respath1,x],dim=1)
        x=self.decodeconv3(x)
        x=self.almostlast(x)   
        return y


def pretrainDenseUnet(model):
    state_dict=torch.load("/root/repo/Chexpert/chexpert/model/output/best_model.pth")
    pretrain=state_dict["model_state_dict"]
    with torch.no_grad():
        model.sequenceLayer.conv0.weight.copy_(pretrain["dense.conv0.weight"])
        for name, module in model.named_children():
            if name.startswith("denseblock"):
                for subname,submodule in getattr(model,name).named_children():
                    layer=getattr(getattr(model,name),subname )
                    layer.conv1.weight.copy_(pretrain["dense.{}.{}.conv1.weight".format(name,subname)])
                    layer.conv2.weight.copy_(pretrain["dense.{}.{}.conv2.weight".format(name,subname)])
