import torch
from torch.utils.data import DataLoader
from datasets.dataset import UrinaryStoneDataset

def get_dataloader(opt):
    trn_dataset = UrinaryStoneDataset(opt, is_Train=True, augmentation=True)
    val_dataset = UrinaryStoneDataset(opt, is_Train=False, augmentation=False)

    train_dataloader = DataLoader(trn_dataset,
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.workers)

    valid_dataloader = DataLoader(val_dataset,
                                  batch_size=opt.batch_size,
                                  shuffle=False,
                                  num_workers=opt.workers)
    
    return train_dataloader, valid_dataloader