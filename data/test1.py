import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import KFold
from batchgenerators.utilities.file_and_folder_operations import *
class CustomDataset(Dataset):
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Generate some example data (you can replace this with your actual data)
        sample = f"idx {index}of length{self.length} "  # Random data for illustration
        return sample

d1=CustomDataset(10)
d2=CustomDataset(50)
dataset = ConcatDataset([d1, d2])
kfold = KFold(n_splits=5, shuffle=True)
# for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
#     print(f'FOLD {fold}')
#     print (train_ids,test_ids)


import pandas as pd

# Create a sample DataFrame
data = {'A': ["a", "a", "b"],
        'B': [4, 5, 6],
        'C': [7, 8, 9]}

train_x = pd.DataFrame(data)
train_x.set_index('A',inplace=True)

# a pandas series can be think as 1 column of dataframe
# print(9 in train_x.loc["b","C"])
print (train_x.loc["b",["C"]])  # it will return 9 only
print (train_x.loc["a","C"])  # it will return  a series
print (type(train_x.loc["b",["C","B"]]))
for x in train_x.loc["a","C"]:
    print (x)  # it will return a single value since dataframe has only 1 column
# Extract values of specific columns ('A' and 'B' in this case)

# print (train_x['A'].values)
# print ( train_x[train_x.keys()])
# selected_columns_values = train_x[train_x.keys()].values

# print(selected_columns_values)


import glob
import numpy as np
import pydicom
path="/root/data/siim-other"
df=pd.read_csv(join(path,"train-rle.csv"))
print (df.columns)
names=df["ImageId"].to_list()

img_glob=join (path,"pneumothorax","dicom-images-train","**/**/*.dcm")
img_ls=glob.glob(img_glob)
img_arr=pydicom.dcmread(img_ls[0])
print (img_arr.pixel_array.shape)
# print (len(glob.glob(img_glob)))#10712

img_list=set([x.split("/")[-1][:-4] for x in glob.glob(img_glob)])
idx=[]

#print (len(df))#11582

for i, row in df.iterrows():
    if row["ImageId"]  in img_list:
        idx.append(i)
# print (len(idx)) #11582

# print (len(df["ImageId"].unique())) #10675


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


data={
    "A":[1,1,3],
    'B':[4,5,6],
    "C":[7,8,9]
}
df=pd.DataFrame(data)
df=df.groupby('A').agg(unique_to_list).reset_index()
print (df)









