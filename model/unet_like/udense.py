import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models.densenet import _Transition, _load_state_dict
import torch.nn.functional as F
import sys
from collections import OrderedDict
from torchvision.models import DenseNet
from collections import OrderedDict
from typing import List
class Encoder(nn.Module):
    def __init__(self,skip_connections,downsample ) -> None:
        
        super().__init__()
        self.skip_connections=skip_connections
        denseNet=torchvision.models.densenet121(weights="DenseNet121_Weights.IMAGENET1K_V1" )
        features = denseNet.features[:-1]          
        self.features=features
        for module in self.features.modules():
            if isinstance(module, nn.AvgPool2d):
                module.register_forward_hook(lambda _, input, output : self.skip_connections.append(input[0]))

    def forward(self,x):
        x=self.features(x)   
        return x

class Decoder(nn.Module):
    def __init__(self,skip_connections,skip_connections_shape,upsample, num_init_features=64,block_config=[6,12,24,16],growth_rate=32) -> None:
        super().__init__()
        self.skip_connections = skip_connections
        self.skip_connections_shape=skip_connections_shape
        self.upsample = upsample
        denseNet=torchvision.models.densenet121(weights="DenseNet121_Weights.IMAGENET1K_V1" )
        features = denseNet.features[4:-2]
        
        num_features=num_init_features
        num_features_list = []
        for i, num_layers in enumerate(block_config):
            # //is floor division
            num_input_features = num_features + num_layers * growth_rate
            num_output_features = num_features // 2
            num_features_list.append((num_input_features, num_output_features))
            num_features = num_input_features // 2
       # num_features_list : [(256, 32), (512, 64), (1024, 128), (1024, 256)]
        
        for i in range (len(features)) :
            if isinstance(features[i], _Transition):
                num_input_features, num_output_features = num_features_list.pop(1)
                features[i]= _TransitionUp(num_input_features, num_output_features,skip_connections,skip_connections_shape=skip_connections_shape)

        self.features=  nn.Sequential(*reversed(features))
        num_input_features, _ = num_features_list.pop(0)      
        if upsample:
            self.features.add_module('upsample0', nn.Upsample(scale_factor=4, mode='bilinear'))
            self.features.add_module('norm0', nn.BatchNorm2d(num_input_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('conv0', nn.Conv2d(num_input_features, num_init_features, kernel_size=1, stride=1, bias=False))
            self.features.add_module('norm1', nn.BatchNorm2d(num_init_features))

    def forward(self, x):
        return self.features(x)        
class _Concatenate(nn.Module):
    def __init__(self, skip_connections):
        super(_Concatenate, self).__init__()
        self.skip_connections = skip_connections
        
    def forward(self, x):
        return torch.cat([x, self.skip_connections.pop()], 1)
# pop will pop the last element
# skip connection :(128,128,128), (256,64,64), (512,32,32)
 #    num_features_list : [(512, 64), (1024, 128), (1024, 256)]
          
class _TransitionUp(nn.Sequential):
    # change tensor input shape from num_input_features to num_output_features, and x2 spatial
    def __init__(self, num_input_features, num_output_features, skip_connections,skip_connections_shape):
        super(_TransitionUp, self).__init__()
       
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, num_output_features * 2,
                                              kernel_size=1, stride=1, bias=False))
        # effect of upsample
        #(C,H,W)=> (C,H* scale factor,W* scale factor)
        self.add_module('upsample', nn.Upsample(scale_factor=2, mode='bilinear'))
        num_feature= num_output_features * 2 + skip_connections_shape.pop(0)
        assert num_feature== num_output_features *4 , "invalid shape"
        self.add_module('cat', _Concatenate(skip_connections))
        self.add_module('norm2', nn.BatchNorm2d(num_feature))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(num_feature, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('cat', _Concatenate(skip_connections))
        self.add_module('norm2', nn.BatchNorm2d(num_output_features * 4))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(num_output_features * 4, num_output_features,
                                          kernel_size=1, stride=1, bias=False))

class DenseUNet(nn.Module):
    def __init__(self, n_classes=1, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0, downsample=True, pretrained_encoder_uri=None, progress=None):
        super(DenseUNet, self).__init__()
        self.skip_connections = []
        num_feature=num_init_features
        self.skip_connections_shape=[]
        for num_layer in block_config[:-1]:
            num_feature=num_feature+growth_rate*num_layer
            self.skip_connections_shape.append(num_feature//2)
            num_feature=num_feature//2
        # self.skip_connections_shape:[128,256,512]
        self.encoder=Encoder(skip_connections=self.skip_connections,downsample=downsample)
       
        self.decoder=Decoder(skip_connections=self.skip_connections,skip_connections_shape=self.skip_connections_shape,upsample=downsample)
        self.classifier = nn.Conv2d(num_init_features, n_classes, kernel_size=1, stride=1, bias=True)         
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)  
        x=self.classifier(x)
           
        return x

def pretrainDenseUnet(model):
    state_dict=torch.load("/root/repo/Chexpert/chexpert/model/output/best_model.pth")
    pretrain=state_dict["model_state_dict"]
    with torch.no_grad():
        model.encoder.features.conv0.weight.copy_(pretrain["dense.conv0.weight"])
        for name, module in model.encoder.features.named_children():
            if name.startswith("denseblock"):
                # print (name)
                for subname,submodule in getattr(model.encoder.features,name).named_children():
                    # print (subname)
                    layer=getattr(getattr(model.encoder.features,name),subname )
                    layer.conv1.weight.copy_(pretrain["dense.{}.{}.conv1.weight".format(name,subname)])
                    layer.conv2.weight.copy_(pretrain["dense.{}.{}.conv2.weight".format(name,subname)])
            if name.startswith("transition"):
                    # print (name)
                    module.conv.weight.copy_(pretrain["dense.{}.conv.weight".format(name)])