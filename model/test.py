from torchvision.models import DenseNet
from torchvision.models import DenseNet
from torchvision.models.densenet import _Transition, _load_state_dict
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
import torchvision
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
        # print (self.features)
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
        #skip_connections : (128, 128, 128 ),  (256, 64, 64), (512, 32, 32)
        # num_features_list [1:]= (512, 64), (1024, 128), (1024, 256)]
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
        # for name, module in self.named_children():
        #     module.__name__=name
        #     module.register_forward_hook(lambda mod,input,output : print (mod.__name__,input[0].shape,output.shape) )

class DenseUNet(nn.Module):
    def __init__(self, n_classes, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0, downsample=True, pretrained_encoder_uri=None, progress=None):
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
        self.softmax = nn.Softmax(dim=1)

             
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)  
        x=self.classifier(x)
           
        return self.softmax(x)
        
dense=DenseUNet(n_classes=5)
x1=torch.rand(4, 1024, 16, 16)
x2=torch.rand(4,3,512,512)

# y=decoder(x1)
# z=encoder(x2)
y=dense(x2)
print (y.shape)


class denseNet(nn.Module):
    def __init__(self,block_config=(6,12,24,16),growth_rate=32, num_init_features=64, bn_size=4):
        super(denseNet, self).__init__()
        num_features=num_init_features
        self.features=nn.Sequential()
        for i,num_layers in enumerate(block_config):
            self.features.add_module(f"dense_block{i+1}",denseBlock(num_layers=num_layers,num_init_features=num_features,growth_rate=growth_rate,bn_size=bn_size))
            num_features= (num_features+growth_rate*num_layers)
            if i != len(block_config)-1:
                self.features.add_module(f"transition{i+1}",Transition(num_input_features=num_features))
            num_features=num_features//2
    def forward(self,x):

        return self.features(x)

class Layer(nn.Module):
    def __init__(self,num_init_features,growth_rate=32,bn_size=4):
        super(Layer,self).__init__()
        self.num_init_features=num_init_features
        self.norm1= nn.BatchNorm2d(num_init_features)
        self.relu=nn.ReLU(inplace=True)
        self.conv1=nn.Conv2d(num_init_features,bn_size*growth_rate,kernel_size=1,stride=1,bias=False)
        self.norm2=nn.BatchNorm2d(bn_size*growth_rate)
        self.conv2=nn.Conv2d(bn_size*growth_rate,growth_rate,kernel_size=3,stride=1,padding=1,bias=False)
        # for name, module in self.named_children():
        #     module.__name__=name
        #     module.register_forward_hook(lambda mod,input,output : print (mod.__name__,input[0].shape,output.shape) )
    def forward(self,x:List[torch.Tensor]):
        x=torch.cat(x,1)
        x=self.norm1(x)
        x=self.relu(x)
        x=self.conv1(x)
        x=self.norm2(x)
        x=self.relu(x)
        x=self.conv2(x)
        return x

class denseBlock(nn.Module):
    def __init__(self,num_layers, num_init_features,growth_rate,bn_size=4):
        super().__init__()
        self.layers=nn.Sequential()
        num_features=num_init_features
        for layer in range(num_layers):
            self.layers.add_module(f"denselayer{layer+1}", Layer(num_init_features=num_features,growth_rate=growth_rate,bn_size=bn_size) )
            num_features+=growth_rate
    def forward(self,x):
        features=[x]       
        for _,layer in self.layers.named_children():
            lay=layer(features)
            features.append(lay)
        y=torch.cat(features,axis=1)
        return y
            
class Transition(nn.Module):
    def __init__(self,num_input_features):
        super().__init__()
        print ("tran",num_input_features)
        self.norm=nn.BatchNorm2d(num_input_features)
        self.relu=nn.ReLU(inplace=True)
        self.conv=nn.Conv2d(num_input_features,num_input_features//2,kernel_size=3,stride=1,padding=1,bias=False)
        self.pool=nn.AvgPool2d(kernel_size=2,stride=2)
        # for name, module in self.named_children():
        #     module.__name__=name
        #     module.register_forward_hook(lambda mod,input,output : print (mod.__name__,input[0].shape,output.shape) )
    def forward(self,x):
        x=self.norm(x)
        x=self.relu(x)
        x=self.conv(x)
        x=self.pool(x)
        return x
# x=torch.rand(1,64,512,512)
# net=denseNet()
# # print (net)
# y=net(x)

