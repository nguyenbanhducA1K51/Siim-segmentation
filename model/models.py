## this file is for experimenting model structure only
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
        
        # for name, module in model.decoder.features.named_children():
        #     if name.startswith("denseblock"):
        #         for subname,submodule in getattr(model.encoder.features,name).named_children():
        #             layer=getattr(getattr(model.encoder.features,name),subname )
        #             layer.conv1.weight.copy_(pretrain["dense.{}.{}.conv1.weight".format(name,subname)])
        #             layer.conv2.weight.copy_(pretrain["dense.{}.{}.conv2.weight".format(name,subname)])
        #      if name.startswith("transition"):
        #             module.conv.weight.copy_(pretrain["dense.{}.conv.weight".format(name)])
# net=DenseUNet(n_classes=2)
# pretrainDenseUnet(net)
# print (net)    
    # class _DenseUNetEncoder(DenseNet):
#     def __init__(self, skip_connections, growth_rate, block_config, num_init_features, bn_size, drop_rate, downsample):
#         super(_DenseUNetEncoder, self).__init__(growth_rate, block_config, num_init_features, bn_size, drop_rate)       
#         self.skip_connections = skip_connections
#         # remove last norm, classifier
#         features = OrderedDict(list(self.features.named_children())[:-1])
        
#         delattr(self, 'classifier')
#         if not downsample:
#             features['conv0'].stride = 1
#             del features['pool0']
#         self.features = nn.Sequential(features)      
#         for module in self.features.modules():
#             if isinstance(module, nn.AvgPool2d):
#                 # purpose of register_forward_hook:

#                 module.register_forward_hook(lambda _, input, output : self.skip_connections.append(input[0]))
#     def forward(self, x):
#         x=self.features(x)
       
#         return x
        
# class _DenseUNetDecoder(DenseNet):
#     def __init__(self, skip_connections, skip_connections_shape, growth_rate, block_config, num_init_features, bn_size, drop_rate, upsample):
#         super(_DenseUNetDecoder, self).__init__(growth_rate, block_config, num_init_features, bn_size, drop_rate)
        
#         self.skip_connections = skip_connections
#         self.upsample = upsample
        
#         # remove conv0, norm0, relu0, pool0, last denseblock, last norm, classifier
#         features = list(self.features.named_children())[4:-2]
#         delattr(self, 'classifier')    
#         # the distance within 1 tuple is 1 dense block
#         # num_features_list : [(256, 32), (512, 64), (1024, 128), (1024, 256)]
       
#         num_features_list=[]
#         num_feature=num_init_features
#         dense_feature= num_feature+ block_config[0] * growth_rate
#         tran_feature=dense_feature//2
#         for i, num_layers in enumerate(block_config[1:]):
#                     num_input_features = tran_feature + num_layers * growth_rate         
#                     num_features_list.append((num_input_features, num_feature))
#                     num_feature=tran_feature
#                     tran_feature = num_input_features // 2

#         for i in range(len(features)):
#             name, module = features[i]
#             if isinstance(module, _Transition):
#                 num_input_features, num_output_features = num_features_list.pop(0)
#                 Tran_layer=_TransitionUp(num_input_features, num_output_features, skip_connections,skip_connections_shape)
#                 features[i] = (name, Tran_layer)

                
#         features.reverse()
#         self.features = nn.Sequential(OrderedDict(features))
#         num_input_features= num_init_features+ block_config[0] * growth_rate  # 256      
#         if upsample:
            
#             modules=nn.Sequential(  

#                      nn.ConvTranspose2d(num_input_features, num_init_features, kernel_size=2, stride=2, bias=False),
#                       nn.BatchNorm2d(num_init_features),
#                       nn.ReLU(inplace=True)          
#                             )

#             self.features+=modules
            

#             # self.features.add_module('norm0', nn.BatchNorm2d(num_input_features))
#             # self.features.add_module('relu0', nn.ReLU(inplace=True))
#             # self.features.add_module('conv0', nn.Conv2d(num_input_features, num_init_features, kernel_size=1, stride=1, bias=False))
#             # self.features.add_module('norm1', nn.BatchNorm2d(num_init_features))

#     def forward(self, x):
#         return self.features(x)
 
# class _Concatenate(nn.Module):
#     def __init__(self, skip_connections):
#         super(_Concatenate, self).__init__()
#         self.skip_connections = skip_connections
        
#     def forward(self, x):
#         x=torch.cat([x, self.skip_connections.pop()], 1)
        
#         return x
# # skip connection :(128,128,128), (256,64,64), (512,32,32)
#  #    num_features_list : [(512, 64), (1024, 128), (1024, 256)]
          
# class _TransitionUp(nn.Sequential):
#     def __init__(self, num_input_features, num_output_features, skip_connections,skip_connections_shape):
#         super(_TransitionUp, self).__init__()
        
#         self.add_module('norm1', nn.BatchNorm2d(num_input_features))
#         self.add_module('relu1', nn.ReLU(inplace=True))
#         self.add_module('conv1', nn.Conv2d(num_input_features, num_output_features * 2,
#                                               kernel_size=1, stride=1, bias=False))
        
#         self.add_module('upsample', nn.Upsample(scale_factor=2, mode='bilinear'))
#         # print ( "tran", num_output_features * 2,skip_connections_shape[0])
#         num_feature= num_output_features * 2 + skip_connections_shape.pop(0)
#         assert num_feature== num_output_features *4 , "invalid shape"
#         self.add_module('cat', _Concatenate(skip_connections))
#         self.add_module('norm2', nn.BatchNorm2d(num_feature))
#         self.add_module('relu2', nn.ReLU(inplace=True))
#         self.add_module('conv2', nn.Conv2d(num_feature, num_output_features,
#                                           kernel_size=1, stride=1, bias=False))
#         # for name, module in self.named_children():
#         #     module.__name__=name
#         #     module.register_forward_hook(lambda mod,input,output : print (mod.__name__,input[0].shape,output.shape) )
       
# class DenseUNet(nn.Module):
#     def __init__(self, n_classes=1, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0, downsample=True, pretrained_encoder_uri=None, progress=None):
#         super(DenseUNet, self).__init__()
#         self.skip_connections = []
#         num_feature=num_init_features
#         self.skip_connections_shape=[]
#         for num_layer in block_config[:-1]:
#             num_feature=num_feature+growth_rate*num_layer
#             self.skip_connections_shape.append(num_feature//2)
#             num_feature=num_feature//2
        
#         self.input_block=nn.Sequential(OrderedDict([
#                 ('conv0', nn.Conv2d(3, 64,kernel_size=7,stride=1,padding=3,bias=False)),            
#                 ('norm0', nn.BatchNorm2d(64)),
#                 ('relu0', nn.ReLU(inplace=True)),
#                 ('conv1', nn.Conv2d(64, 64,kernel_size=3,stride=1,padding=1,bias=False)),            
#                 ('norm1', nn.BatchNorm2d(64)),
#                 ('relu1', nn.ReLU(inplace=True)),
#                 ('conv2', nn.Conv2d(64, 64,kernel_size=3,stride=1,padding=1,bias=False)),            
#                 ('norm2', nn.BatchNorm2d(64)),
#                 ('relu2', nn.ReLU(inplace=True)),
#                 ('conv3', nn.Conv2d(64, 64,kernel_size=1,stride=1,bias=False)),            
#                 ('norm3', nn.BatchNorm2d(64)),
#                 ('relu3', nn.ReLU(inplace=True)),
#                 ('pool', nn.MaxPool2d(kernel_size=2,stride=2))
               
# ]))
#         self.eventual_up=nn.Sequential(OrderedDict([
   
#                 ('conv1', nn.ConvTranspose2d(num_init_features*2, num_init_features, kernel_size=2, stride=2, bias=False) )  ,
#                 ('norm1', nn.BatchNorm2d(num_init_features)),
#                 ('relu',  nn.ReLU(inplace=True) )                         
                            
#         ]))
#         self.encoder = _DenseUNetEncoder(self.skip_connections, growth_rate, block_config, num_init_features, bn_size, drop_rate, downsample)
    
#         self.decoder = _DenseUNetDecoder(self.skip_connections, self.skip_connections_shape,growth_rate, block_config, num_init_features, bn_size, drop_rate, downsample)

#         self.classifier = nn.Conv2d(num_init_features, n_classes, kernel_size=1, stride=1, bias=True)
        
#         self.encoder._load_state_dict = self.encoder.load_state_dict
#         self.encoder.load_state_dict = lambda state_dict : self.encoder._load_state_dict(state_dict, strict=False)
#         if pretrained_encoder_uri:
#             _load_state_dict(self.encoder, str(pretrained_encoder_uri), progress)
#         self.encoder.load_state_dict = lambda state_dict : self.encoder._load_state_dict(state_dict, strict=True)

#     def forward(self, x):
#         skip=self.input_block(x)
#         x = self.encoder(x)
      
#         x = self.decoder(x)
#         x=torch.cat([x,skip],1)
#         x=self.eventual_up(x)
#         y = self.classifier(x)
#         return y