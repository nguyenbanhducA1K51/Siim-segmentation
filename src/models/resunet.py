import torch.nn as nn
from torchvision import models
import torch
import sys
sys.path.append("/root/repo/Siim-segmentation/src/")
from unet import ResnetSuperVision
class DecoderBlock(nn.Module):
    def __init__(self, skip_channel,in_channel,middle_channel,out_channel):
        super().__init__()
        self.upsample=nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channel,middle_channel,3,padding=1)

        )
        self.conv=nn.Sequential(
            nn.Conv2d(middle_channel +skip_channel, out_channel, 3, padding=1),
              nn.ReLU(inplace=True)
        )
    def forward(self,skip,x):
        x=self.upsample(x)
        return self.conv( torch.cat([ x,skip],1))
class ResUnet(nn.Module):
    def __init__(self,out_channel):
        super().__init__()
        
        self.backbone=models.resnet34(pretrained=True)
        
        self.encoder0=nn.Sequential(self.backbone.conv1,
                                   self.backbone.bn1,
                                   self.backbone.relu
                                   
                                   )
        self.encoder1=nn.Sequential(
            self.backbone.maxpool,
            self.backbone.layer1
        )
        self.encoder2=self.backbone.layer2
        self.encoder3=self.backbone.layer3
        self.encoder4=self.backbone.layer4
        self.encode=[

            self.encoder0,
            self.encoder1,
            self.encoder2,
            self.encoder3,
            self.encoder4
        ]
        self.decode4= DecoderBlock(256,512,512,256)
        self.decode3=DecoderBlock(128,256,256, 256)  

        self.decode2=DecoderBlock(64,256,128, 128)

        self.decode1=DecoderBlock(64,128,64,64)
        self.decode=[ self.decode4,self.decode3,self.decode2,self.decode1]
        self.final_up=nn.Sequential( 

            nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64,out_channel,3,padding=1)  
                      )     )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 1)


    def forward(self,x):
        encode_block=[]
        for block in self.encode:
            x=block(x)
            encode_block.append(x.clone())

        cls=self.avgpool(x).view(x.shape[0],-1)
        cls=self.fc(cls)

        for  i,block in enumerate(self.decode):

            x= block( encode_block[-i-2], x )
        x=self.final_up(x)
        return x,cls
if __name__=="__main__":
    net=ResUnet(1).to("cuda")
    # x=torch.rand(2,3,256,256).to("cuda")
    x=torch.rand(16,3,512,512).to("cuda")
    net2=ResnetSuperVision(1, backbone_arch='resnet34').to("cuda")
    y1,y2=net(x)
    z1,z2=net2(x)
    print (y1.shape,y2.shape,z1.shape,z2.shape)



