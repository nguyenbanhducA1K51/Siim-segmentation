
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import sys
from collections import OrderedDict
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels,mid_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        x=self.conv(x)     
        return x
class ConvBlock(nn.Module):
    def __init__(self,in_channel,out_channel,padding=1, kernel_size=3, stride=1 ):
        super().__init__()
        self.conv=nn.Conv2d(in_channel,out_channel,kernel_size=kernel_size,stride=stride,padding=padding)
        self.batchnorm=nn.BatchNorm2d(out_channel)
        self.relu=nn.ReLU()
    def forward(self,x):
        x=self.conv(x)
        x=self.batchnorm(x)
        x=self.relu(x)
        return x 

class Upblock(nn.Module):
    def __init__(self,in_channel,out_channel,up_in_channel,up_out_channel,kernel_size=2,stride=2) -> None:
        super().__init__()
        self.upsample=nn.ConvTranspose2d(up_in_channel,up_out_channel,kernel_size=kernel_size,stride=stride)
        self.conv_block1=ConvBlock(in_channel=in_channel,out_channel=out_channel)
        self.conv_block2=ConvBlock(in_channel=out_channel,out_channel=out_channel)
    def forward(self,x_up,x_down):
        x=self.upsample(x_up)
        x=torch.cat((x,x_down),1)

        x=self.conv_block1(x)
        x=self.conv_block2(x)
        return x

class Bridge(nn.Module):
    def __init__(self,in_channel,out_channel,padding=1, kernel_size=3, stride=1 ):
        self. conv=nn.ConvBlock
class DenseUNet (nn.Module):
    def __init__ (self,imageSize,numclass):
        super(DenseUNet, self).__init__()
        assert imageSize%32==0 ,"input image must be divisible by 32"
        self.numclass=numclass
        self.imageSize=imageSize
        denseNet=models.densenet121(weights="DenseNet121_Weights.IMAGENET1K_V1" )
        #(3,512,512)
        self.inputblock =nn.Sequential( OrderedDict ([ 
                ("conv0", nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) ),
                ( "norm0", nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) ),
                ("relu0", nn.ReLU(inplace=True) ),
                ( "pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False) ) ]) )
        #(64,128,128)
        self.denseblock1=denseNet.features.denseblock1
        #256,128,128
        self.transition1=denseNet.features.transition1
        #128,64,64
        self.denseblock2=denseNet.features.denseblock2
        #512,64,64
        self.transition2=denseNet.features.transition2
        #256,32,32
        self.denseblock3=denseNet.features.denseblock3
        #1024,32,32
        self.transition3=denseNet.features.transition3
        # 512,16,16
        self.denseblock4=denseNet.features.denseblock4
        # 1024,16,16
        self.bottle= nn.Sequential( 
            OrderedDict ([ 
                ("conv0", nn.Conv2d(1024, 2048, kernel_size=3 ,stride=2, padding=1, bias=False) ),
                ( "norm0", nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) ),
                ("relu0", nn.ReLU(inplace=True) ),
                ]) 

        )
       
        self.norm5=denseNet.features.norm5

        self.upblock1=Upblock(in_channel=1024+1024,out_channel=1024,up_in_channel=2048,up_out_channel=1024)
         # up (2048,8,8) to (1024,16,16)+(1024,16,16) +conv= (1024,16,16)
        self.upblock2=Upblock(in_channel=1024+1024,out_channel=512,up_in_channel=1024,up_out_channel=1024)
        # up (1024,16,16) to (1024,32,32)+(1024,32,32) +conv= (512,32,32)
        self.upblock3=Upblock(in_channel=512+512,out_channel=256,up_in_channel=512,up_out_channel=512)
        #  up (512,32,32) to (512,64,64) +(512.64,64)+conv=256,64,64
        self.upblock4=Upblock(in_channel=256+256,out_channel=128,up_in_channel=256,up_out_channel=256)
        #up(256,64,64) to (256,128,128) + (256,128,128) +conv= (128,128,128)
        self.upblock5=Upblock(in_channel=3+128,out_channel=numclass,up_in_channel=128,up_out_channel=128,kernel_size=4,stride=4)
        # up 128,128,128 to 128,512,512 + 3,512,512 +conv =class,512,512     
        self.sigmoid=nn.Sigmoid()            
    def forward(self,x):
        #encoder
        idenity=x
        x=self.inputblock(x)
        respath1=self.denseblock1(x)
        x=self.transition1(respath1)
        respath2=self.denseblock2(x)
        x=self.transition2(respath2)
        respath3=self.denseblock3(x)
        x=self.transition3(respath3)
        respath4=self.denseblock4(x)
        x=self.norm5(respath4)  
        bottle=self.bottle(x)
        x=self.upblock1(bottle,respath4)
        x=self.upblock2(x,respath3)
        x=self.upblock3(x,respath2)
        x=self.upblock4(x,respath1)
        x=self.upblock5(x,idenity)
           
        return x


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

net=DenseUNet(imageSize=512,numclass=4)
a=torch.rand(5,3,512,512)
y=net(a)
print (y.shape)