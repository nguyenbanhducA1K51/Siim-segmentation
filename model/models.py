
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import sys
from collections import OrderedDict

# should train this unet to have pretrain weight
class UNet(nn.Module):
   
    # Unet model require down width to be divisible by 16 
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.block_down1 = self.unet_block(3, 64)
        self.block_down2 = self.unet_block(64, 128)
        self.block_down3 = self.unet_block(128, 256)
        self.block_down4 = self.unet_block(256, 512)
        self.block_neck = self.unet_block(512, 1024)
        self.block_up1 = self.unet_block(1024+512, 512)
        self.block_up2 = self.unet_block(256+512, 256)
        self.block_up3 = self.unet_block(128+256, 128)
        self.block_up4 = self.unet_block(128+64, 64)
        if self.n_classes==2:

            self.conv_cls = nn.Conv2d(64, 1, 1) # -> (B, n_class, H, W)
        else:
            self.conv_cls = nn.Conv2d(64, self.n_classes, 1)

    
    def forward(self, x):
        # (B, C, H, W)
        x1 = self.block_down1(x)
        x = self.downsample(x1)
        x2 = self.block_down2(x)
        x = self.downsample(x2)
        x3 = self.block_down3(x)
        x = self.downsample(x3)
        x4 = self.block_down4(x)
        x = self.downsample(x4)

        x = self.block_neck(x)

        x = torch.cat([x4, self.upsample(x)], dim=1)
        x = self.block_up1(x)
        x = torch.cat([x3, self.upsample(x)], dim=1)
        x = self.block_up2(x)
        x = torch.cat([x2, self.upsample(x)], dim=1)
        x = self.block_up3(x)
        x = torch.cat([x1, self.upsample(x)], dim=1)
        x = self.block_up4(x)

        x = self.conv_cls(x)
        return x
    def unet_block(self,in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.ReLU()
        )     
#image size: 1024*1024
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
        # print ("1",x.shape)
        x=self.denseblock1(respath1)
        #256, 128, 128
        # print ("res1",respath1.shape)
        respath2=self.transition1(x)
        #128, 64, 64
        # print ("2",x.shape)
        x=self.denseblock2(respath2)
        #512, 64, 64
        # print ("res2",respath2.shape)
        respath3=self.transition2(x)
        #256, 32, 32]
        # print ("3",x.shape)
        x=self.denseblock3(respath3)
        # 1024, 32, 32
        # print ("res3",respath3.shape)
        respath4=self.transition3(x)
        #512,16,16
        # print ("4",x.shape)
        x=self.denseblock4(respath4)
        #1024,16,16
        # print ("res4",respath4.shape)
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
        y=self.sigmoid(x)
       
        return y


# print (models.densenet121())
# class newnet(nn.Module):
#     def __init__(self):
#         super(newnet,self).__init__()
#         self.newclass1=nn.Linear(5,7)
#         self.newclass2=nn.Linear(3,4)
#     def forward(self,x):
#         x=self. newclass1(x)
#         return (x)
# class net(nn.Module):
#     def __init__(self):
#         super(net,self).__init__()
#         self.class1=nn.Linear(5,7)
#         self.class2=nn.Linear(3,4)
#     def forward(self,x):
#         x=self.class1(x)
#         return (x)
# model= newnet()
# # print (model)
# torch.save({
                
#                 'model_state_dict': model.state_dict(),
               
#                 }, "/root/repo/Siim-segmentation/model/output/test.pth"
#                 )
# state_dict=torch.load("/root/repo/Siim-segmentation/model/output/test.pth")
# # print (param["model_state_dict"]["class1.weight"])
# newmodel=net()
# with torch.no_grad():
#     print ("before :{}".format(newmodel.class1.weight))
#     print ("after : {}".format(newmodel.class1.weight.copy_(state_dict["model_state_dict"]['newclass1.weight'])))