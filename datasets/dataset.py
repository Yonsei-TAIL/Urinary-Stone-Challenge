import torch
from torch.utils.data import Dataset

import cv2
import os
import numpy as np
from glob import glob
import SimpleITK as sitk

from utils.transforms import image_windowing, image_minmax, mask_binarization, augment_imgs_and_masks

class UrinaryStoneDataset(Dataset):
    def __init__(self, opt, is_Train=True, augmentation=False):
        super(UrinaryStoneDataset, self).__init__()

        self.dcm_list = glob(os.path.join(opt.data_root, 'Train' if is_Train else 'Valid', 'DCM', '*.dcm'))
        self.len = len(self.dcm_list)

        self.augmentation = augmentation
        self.opt = opt

        self.is_Train = is_Train

    def __getitem__(self, index):
        # Load Image and Mask
        dcm_path = self.dcm_list[index]
        mask_path = dcm_path.replace('DCM', 'Label').replace('.dcm', '.png')

        img_sitk = sitk.ReadImage(dcm_path)
        img = sitk.GetArrayFromImage(img_sitk)[0]
        mask = cv2.imread(mask_path, 0)

        # HU Windowing
        img = image_windowing(img, self.opt.w_min, self.opt.w_max)

        # MINMAX to [0, 255] and Resize
        img = image_minmax(img)
        img = cv2.resize(img, (self.opt.input_size, self.opt.input_size))
        mask = cv2.resize(mask, (self.opt.input_size, self.opt.input_size))

        # MINMAX to [0, 1]
        img = img / 255.

        # Mask Binarization (0 or 1)
        mask = mask_binarization(mask)

        # Add channel axis
        img = img[None, ...].astype(np.float32)
        mask = mask[None, ...].astype(np.float32)
                
        # Augmentation
        if self.augmentation:
            img, mask = augment_imgs_and_masks(img, mask, self.opt.rot_factor, self.opt.scale_factor, self.opt.trans_factor, self.opt.flip)

        return img, mask
        
    def __len__(self):
        return self.len