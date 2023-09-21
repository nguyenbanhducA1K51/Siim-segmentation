import pandas as pd 
import random 
import os
import glob
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import random
import shutil
from PIL import Image
import ast
from sklearn.model_selection import StratifiedKFold
def preprocess(cfg):
    png_save_path=cfg.path.png_save_path
    if os.path.exists(png_save_path):
        shutil.rmtree(png_save_path)

    if os.path.exists(os.path.join(png_save_path,"img")):
        shutil.rmtree(os.path.join(png_save_path,"img"))
    
    if os.path.exists(os.path.join(png_save_path,"msk")):
        shutil.rmtree(os.path.join(png_save_path,"msk"))

    if os.path.exists(os.path.join(png_save_path,"inference")):
        shutil.rmtree(os.path.join(png_save_path,"inference"))
    os.makedirs(png_save_path)
    os.makedirs(os.path.join(png_save_path,"img"))
    os.makedirs(os.path.join(png_save_path,"msk"))
    os.makedirs(os.path.join(png_save_path,"inference"))

    
    img_dir=os.path.join(png_save_path,"img")
    msk_dir=os.path.join(png_save_path,"msk")
    infer_dir=os.path.join(png_save_path,"inference")
    train_dicom_path=cfg.path.train_dicom_path
    infer_dicom_path=cfg.path.infer_dicom_path

    train_rle_df=pd.read_csv(cfg.path.train_rle_csv)

    # damn, the column EncodedPixels has extra space ahead in it name
    train_rle_df =  train_rle_df.rename(columns={' EncodedPixels': 'EncodedPixels'})
    test_ratio=1/10
    val_ratio=1/10
    print (f"Original number of train samples {len(train_rle_df)}")
    train_rle_df=train_rle_df.groupby('ImageId').agg(unique_to_list).reset_index(drop=True)
    def stripping (y):
        # y=ast.literal_eval(y)  
        return [x.strip() for x in y]
    train_rle_df["EncodedPixels"]=train_rle_df["EncodedPixels"].apply(stripping)
    def has_pneumo(y ):
        if "-1" in y:
            return 0
        return 1
    train_rle_df["has_pneumo"]=train_rle_df["EncodedPixels"].apply(has_pneumo)
    print (f"Unique number of train samples {len(train_rle_df)}")
    test_val_idx=random.sample(range(len(train_rle_df)),int( (test_ratio+val_ratio)*len(train_rle_df)))
    test_idx=test_val_idx[: int( (test_ratio)*len(train_rle_df))]
    val_idx=test_val_idx[int( (test_ratio)*len(train_rle_df)):]
    train_idx=[x for x in range(len(train_rle_df)) if x not in  test_val_idx]

    train_csv_png_path=os.path.join(png_save_path,'train.csv')
    val_csv_png_path=os.path.join(png_save_path,'val.csv')
    test_csv_png_path=os.path.join(png_save_path,'test.csv')
  
    train_rle_df.iloc[test_idx].copy().to_csv(test_csv_png_path, index=False)
    train_rle_df.iloc[val_idx].copy().to_csv(val_csv_png_path,index=False)
    train_rle_df.iloc[train_idx].copy().to_csv(train_csv_png_path,index=False)
    relocate_dicom(train_rle_df, train_dicom_path ,img_dir,msk_dir,cfg.image.size)
    move_dicom_inference(  infer_dicom_path,infer_dir)

    train=train_csv_png_path
    val=val_csv_png_path
    path=os.path.join(png_save_path,'k_fold.csv')

    df1=pd.read_csv(train)
    df2=pd.read_csv(val)
    df = pd.concat([df1, df2], axis=0, ignore_index=True).reset_index(drop=True)

    indices=range(len(df))
    n_splits = 5

    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for i,(train_idx, test_idx) in enumerate( stratified_kfold.split(indices, df["has_pneumo"].to_list())):
        df.loc[test_idx,"fold"]=str(i+1)
    df.to_csv(path,index=False)



def move_dicom_inference(infer_dicom_path,infer_dir):
    dicom_paths=glob.glob(os.path.join(infer_dicom_path,"**/**/*.dcm"))
    assert (len(dicom_paths)>0), "empty"
    for idx,dicom_path in enumerate(dicom_paths):
      
        img_base= dicom_path.split("/")[-1][:-4]
        print (f'move infer image {img_base}')
        img=pydicom.dcmread(dicom_path).pixel_array   
        image = Image.fromarray(img, mode='L')
        image.save(os.path.join(infer_dir,f"{img_base}.png"))

def relocate_dicom(df,train_dicom_path ,img_dir,msk_dir,image_size=1024):
    num_visualize=10

    dicom_paths=glob.glob(os.path.join(train_dicom_path,"**/**/*.dcm"))
    basename_set=set(df["ImageId"])
    num_label_img=0   
    #image range is [0,255]
    sample= pydicom.dcmread(dicom_paths[0]).pixel_array
    print (f"original image shape {sample.shape}")

    visualize_ls=[]
    for idx,dicom_path in enumerate(dicom_paths):
        
        img=pydicom.dcmread(dicom_path).pixel_array   
        msk=np.zeros((image_size,image_size))
        img_base= dicom_path.split("/")[-1][:-4]
        print (f"process {idx}/{len(dicom_paths)}")
        if  img_base in basename_set:
            num_label_img+=1
            # remember   mask  is transpose here !
            msk=read_mask(df.loc[df["ImageId"]==img_base]["EncodedPixels"], image_size).T
            if np.sum(msk)>0 and len(visualize_ls)<num_visualize:
                visualize_ls.append(tuple([img,msk]))
        else:
            print (f"Image {dicom_path} is not labeled, assume healthy patient")
     
        cv2.imwrite(os.path.join(img_dir,f"{img_base}.png"), img)
        cv2.imwrite(os.path.join(msk_dir,f"{img_base}.png"), msk)
       
    visualize(visualize_ls)
    print ("Finish")

def visualize(visualize_ls):
    random_integer = random.randint(1, 100)
    save_path=f"/root/repo/Siim-segmentation/data/visualize.png"
    fig,ax=plt.subplots(len(visualize_ls),3,figsize=(30,30))
    plt.tight_layout()
    for idx,(img,msk) in enumerate(visualize_ls):
        
        ax[idx,0].imshow(img,cmap="bone")
        ax[idx,0].set_title(" image ")
        ax[idx,1].imshow(msk,cmap="bone")
        ax[idx,1].set_title(" mask ")

        ax[idx,2].imshow(img,cmap="bone")
        ax[idx,2].imshow(msk,alpha=0.5, cmap="Reds")
    plt.show()
    plt.savefig(save_path)

    
def read_mask(msk_ls,image_size=1024):
    #msk_ls is a dataframe series with 1 element
    msk_ls=msk_ls.item()
    msk=np.zeros((image_size,image_size))

    if  "-1" in msk_ls or " -1"in msk_ls:
        return msk
    else:
        for mask in msk_ls:

            msk+=rle2mask(mask,image_size,image_size)
    msk[msk>255]=255
    msk[msk>0]=255

    return msk             

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

    
if __name__=="__main__":
    from easydict import EasyDict as edict
    import os,json
    import cv2
    PROJECT_PATH=os.environ("PROJECT_ROOT")
    cfg_path=os.path.join(PROJECT_PATH, "config/config.json")
    with open(cfg_path) as f:
        cfg = edict(json.load(f))
    preprocess(cfg)
 

