
SIIM-ACR Pneumothorax segmentation

## Purpose
This repository attempts to perform segmentation task on Pneumothorax disease from chest x-ray image
## Dataset
Pneumothorax can be caused by a blunt chest injury, damage from underlying lung disease, or most horrifying—it may occur for no obvious reason at all. On some occasions, a collapsed lung can be a life-threatening event.
Pneumothorax is usually diagnosed by a radiologist on a chest x-ray, and can sometimes be very difficult to confirm. To learn more about dataset visit kaggle website
https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation. Since this dataset is no longer on the official competition, you can download the dataset at this site: 
https://www.kaggle.com/datasets/jesperdramsch/siim-acr-pneumothorax-segmentation-data
## Installation
Clone the repository
```bash
git clone https://github.com/nguyenbanhducA1K51/Siim-segmentation.git
```

cd to the repo and use the package manager [pip](https://pip.pypa.io/en/stable/) to install libary and package in file requirements.txt (recommend install in conda environment).

```bash
pip install -r requirements.txt
```

## Usage

- Open the ../config/config.json file and change the value of following variable:
"repo_path": path to the cloned repo

"train_dicom_path": path to the train dicome file, for example "/root/data/siim-other/pneumothorax/dicom-images-train"

"infer_dicom_path": path to the infer dicome file,

"train_rle_csv": path to the train-rle csv file, for exmample "/root/data/siim-other/pneumothorax/train-rle.csv"

"png_save_path": path that will convert dicom files to png files 

- First , run the preprocess file at "..src/preprocess.py" 
```bash
python3 ./src/preprocess.py
```

- Run the training pipeline by
```bash
bash /scripts/train.sh
```




