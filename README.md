# Urinary Stone Challenge
This repository is the 3rd place solution for the urinary stone segmentation of [2020 AI Smart Challenge](http://wonmcrc.org/).

**NOTE** : It is unable to run this code on your local machine because the challenge was private. You have to customize the [dataset.py](https://github.com/Yonsei-TAIL/Urinary-Stone-Challenge/blob/main/datasets/dataset.py) to run training code with your own dataset and specify the directory on [data_root argument](https://github.com/Yonsei-TAIL/Urinary-Stone-Challenge/blob/main/datasets/dataset.py).

## Getting Started
Requirements:
- Python 3 (code has been tested on Python 3.7)
- PyTorch (code tested with 1.6.0)
- CUDA and cuDNN
- Python pacakges : tqdm, opencv-python (4.4.0), SimpleITK (1.2.4), scipy (1.2.1), imgaug (0.4.0), tensorboardX

Structure:
- ```datasets/```: data loading code
- ```network/```: network architecture definitions
- ```options/```: argument parser options
- ```utils/```: image processing code, miscellaneous helper functions, and training/evaluation core code
- ```train.py/```: code for model training
- ```test.py/```: code for model evaluation
- ```main.ipynb/``` : our main source code on jupyter notebook ([Notebook](https://github.com/Yonsei-TAIL/Urinary-Stone-Challenge/blob/main/main.ipynb))

#### Dataset
The dataset architecture must be as below.
```
DataSet
└─── train
│    └─── DICOM
│    │    │   case_001.dcm
│    │    │   case_002.dcm
│    │    │   case_003.dcm
│    │    │   ...
│    │    │
│    └─── Label
│    │    │   case_001.png
│    │    │   case_002.png
│    │    │   case_003.png
│    │    │   ...
│    │    │
└─── Test
     └─── DICOM
     
```


#### Training and Testing
- To train a network, call: ```python train.py --batch_size 8```
- To evaluate a network after training, call: ```python evaluate.py --resume trained_weights.pth```

#### Performance (Modified UNet 2D)
|    Model     | Dice |  mIoU  |
| :----------: | :--: | :----: |
| Modified-Unet|      | 0.7243 |


#### Pre-trained Models
- Modified-UNet : Not released yet.
