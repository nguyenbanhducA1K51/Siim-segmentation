
import torch.nn as nn
import torch
# should train this unet to have pretrain weight
class UNet(nn.Module):
   
    # Unet model require input width to be divisible by 16 
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

