import os
import cv2
import glob
import pydicom
from tqdm import tqdm_notebook as tqdm
import zipfile
import io
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import exposure
import sys
from batchgenerators.utilities.file_and_folder_operations import *
# path= "/root/data/siim"
# csv_path="/root/data/siim/train-rle.csv"

# names_list=glob.glob(join( path,"dicom-images-train", '**/**/*.dcm'))
# new_names_list=[ os.path.splitext(x.split("/")[-1] )[0] for x in names_list]
# print (new_names_list[:4])
# names=set(new_names_list)

# labeled_images=[]
# print (len(names))
# for idx,row in df.iterrows():
#     print (row["ImageId"])
#     if str(row["ImageId"]) in names:
#         labeled_images.append(idx)
#     if idx==5:
#         break
    
# print (len(labeled_images))


# x=set(["abc","def"])
# print ("abc"in x)



# print (df.info())
# print (df.head()["ImageId"].to_list())

# print (len(glob.glob("/root/data/siim-png/train/*")))

# print (len(glob.glob("/root/data/siim-png/masks/*")))

path= "/root/data/siim-other"
train_glob = '/root/data/siim-other/pneumothorax/dicom-images-train/*/*/*.dcm'
test_glob = '/root/data/siim-other/pneumothorax/dicom-images-test/*/*/*.dcm'
train_fns = sorted(glob.glob(train_glob))
test_fns = sorted(glob.glob(test_glob))
print (len(train_fns),len(test_fns))


train_glob = '/root/data/siim-other/dicom-images-train/*/*/*.dcm'
test_glob = '/root/data/siim-other/dicom-images-test/*/*/*.dcm'
train_fns2 = sorted(glob.glob(train_glob))
test_fns2 = sorted(glob.glob(test_glob))
print (len(train_fns2),len(test_fns2))
print (train_fns[0])
print (train_fns2[0])

N=5
fig, ax = plt.subplots(nrows=N, ncols=2, sharey=True, figsize=(30,30))

for i in range (N):
    dataset1=pydicom.dcmread(train_fns[i])
    dataset2=pydicom.dcmread(train_fns2[i])
    ax[i,0].imshow(dataset1.pixel_array, cmap=plt.cm.bone)
    ax[i,1].imshow(dataset2.pixel_array, cmap=plt.cm.bone)
    plt.show()
    plt.savefig('/root/repo/Siim-segmentation/data/visualize2.png')



