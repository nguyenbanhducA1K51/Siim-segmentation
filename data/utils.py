import albumentations as A
import numpy as np
from typing import Union, Tuple, List,Literal

def unique_to_list(series):
    return list(series.unique())


def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)

def load_transform(cfg,mode:Literal["train","val","test"]="train", train_mode:Literal["default","progressive"]="default") :     
    factor=0.05
    if train_mode=="default":           
        img_size=cfg.image.size
    elif train_mode=="progressive":
        img_size=cfg.image.size//2
    else :
        raise RuntimeError("invalid train mode")
    
    ceil=int (img_size*(1+factor) )
    floor=int (img_size*(1-factor) )
    def norm (mask,*args,**kargs):
        mask=mask/255
        mask.astype(float) 
        return mask
    def stackChannel(image,*args,**kargs):
        image=np.expand_dims(image,axis=0)
        return np.repeat(image,3,axis=0)

    train_transform = A.Compose([                     
                    A.ShiftScaleRotate( scale_limit =(-0.2, 0.2) ,rotate_limit=(-10,10)),
                    A.RandomResizedCrop(height=img_size,width=img_size,scale=(0.9, 1.0),ratio=(0.75, 1.3333333333333333)),

                    A.HorizontalFlip(),
                    A.Lambda(image=stackChannel, mask=norm),              
                                ])

    test_transform=A.Compose([  
                    A.Resize(height=img_size,width=img_size),
                    A.Lambda(image=stackChannel, mask=norm),
                                    ])  
    if mode=="train":
          return train_transform
    elif mode=="test" or mode=="val":
          return test_transform
    else:
          raise RuntimeError("invalid mode in transform file ")