# DGAUNet dual-stream mask guided attention U-Net
model：
![image](images/1.png)
# Environments
* Python 3.9
* Pytorch 2.2.1
* GPU RTX2080Ti
# Datasets
The dataset will be uploaded after the paper is published  
Please put the GCPS dataset as the following architecture.
```
├── data
    ├── GCPS
        ├── images
        |       ├── 0
        |           ├── 1.png
        |           ├── 2.png
        |           ├── ...
        |
        ├── masks
        |       ├── 0
        |           ├── 1.png
        |           ├── 2.png
        |           ├── ...
        |
├── src
├── train.py
├── split.py
├── ...
```
#Training
```
python DGAUNet_train.py --base_dir ./data/GCPS --train_file_dir GCPS_train.txt --val_file_dir GCPS_val.txt --base_lr 0.01 --epoch 150 --batch_size 8
```

