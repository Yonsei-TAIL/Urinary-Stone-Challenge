# Urinary Stone Challenge

## Getting Started
This repository provides everything necessary to train and evaluate a urinary stone segmentation model.\
The baseline network is [Modified-UNet](https://github.com/pykao/Modified-3D-UNet-Pytorch).

Requirements:
- Python 3 (code has been tested on Python 3.5.6)
- PyTorch (code tested with 1.1.0)
- CUDA and cuDNN (tested with Cuda 10.0)
- Python pacakges : tqdm, opencv-python (4.1), SimpleITK (1.2.0), scipy (1.2.1), imgaug (0.4.0), medpy (0.4.0)

Structure:
- ```datasets/```: data loading code
- ```network/```: network architecture definitions
- ```options/```: argument parser options
- ```utils/```: image processing code, miscellaneous helper functions, and training/evaluation core code
- ```train.py/```: code for model training
- ```test.py/```: code for model evaluation

#### Dataset
The dataset architecture must be as below.
```
DataSet
└─── Train
│    └─── DCM
│    │    │   1.dcm
│    │    │   2.dcm
│    │    │   3.dcm
│    │    │   ...
│    │    │
│    └─── Label
│    │    │   1.png
│    │    │   2.png
│    │    │   3.png
│    │    │   ...
│    │    │
└─── Valid
     └─── DCM
     └─── Label
└─── Test
     └─── DCM
     
```


#### Training and Testing
- To train a network, call: ```python train.py --batch_size 1 --in_dim 2 --in_res 140```

- To evaluate a network after training, call: ```python evaluate.py --in_dim 2 --resume trained_weights.pth```

- To inference a network, call: ```python inference.py --in_dim 2 --resume trained_weights.pth```\

#### Performance (Modified UNet 2D)
|    Model     | Dice | mIoU |
| :----------: | :--: | :--: |
| Modified-Unet|      |      |


#### Pre-trained Models
- Modified-UNet : Not released yet.