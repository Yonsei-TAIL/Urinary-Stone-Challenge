import os
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from medpy.metric.binary import sensitivity, specificity, dc, hd95

import torch
from torch.autograd import Variable

from utils import AverageMeter
from utils.metrics import DiceCoef

def train(net, dataset_trn, optimizer, criterion, epoch, opt):
    print("Start Training...")
    net.train()

    losses, total_dices = AverageMeter(), AverageMeter()

    for it, (img, mask) in enumerate(dataset_trn):
        # Optimizer
        optimizer.zero_grad()

        # Load Data
        img, mask = torch.Tensor(img).float(), torch.Tensor(mask).float()
        if opt.use_gpu:
            img, mask = img.cuda(non_blocking=True), mask.cuda(non_blocking=True)

        # Predict
        pred = net(img)

        # Loss Calculation
        loss = criterion(pred, mask)

        # Backward and step
        loss.backward()
        optimizer.step()

        # Calculation Dice Coef Score
        dice = DiceCoef(return_score_per_channel=False)(pred.sigmoid(), mask)
        total_dices.update(dice.item(), img.size(0))

        # Stack Results
        losses.update(loss.item(), img.size(0))

        if (it==0) or (it+1) % 10 == 0:
            print('Epoch[%3d/%3d] | Iter[%3d/%3d] | Loss %.4f | Dice %.4f'
                % (epoch+1, opt.max_epoch, it+1, len(dataset_trn), losses.avg, total_dices.avg))

    print(">>> Epoch[%3d/%3d] | Training Loss : %.4f | Dice %.4f\n"
        % (epoch+1, opt.max_epoch, losses.avg, total_dices.avg))


def validate(dataset_val, net, criterion, epoch, opt, best_dice, best_epoch):
    print("Start Evaluation...")
    net.eval()

    # Result containers
    losses, total_dices = AverageMeter(), AverageMeter()

    for it, (img, mask) in enumerate(dataset_val):
        # Load Data
        img, mask = torch.Tensor(img).float(), torch.Tensor(mask).float()
        if opt.use_gpu:
            img, mask = img.cuda(non_blocking=True), mask.cuda(non_blocking=True)

        # Predict
        pred = net(img)

        # Loss Calculation
        loss = criterion(pred, mask)

        # Calculation Dice Coef Score
        dice = DiceCoef(return_score_per_channel=False)(pred.sigmoid(), mask)
        total_dices.update(dice.item(), img.size(0))

        # Stack Results
        losses.update(loss.item(), img.size(0))

        if (it==0) or (it+1) % 10 == 0:
            print('Epoch[%3d/%3d] | Iter[%3d/%3d] | Loss %.4f | Dice %.4f'
                % (epoch+1, opt.max_epoch, it+1, len(dataset_val), losses.avg, total_dices.avg))

    print(">>> Epoch[%3d/%3d] | Test Loss : %.4f | Dice %.4f"
        % (epoch+1, opt.max_epoch, losses.avg, total_dices.avg))

    # Update Result
    if total_dices.avg > best_dice:
        print('Best Score Updated...')
        best_dice = total_dices.avg
        best_epoch = epoch

        # Remove previous weights pth files
        for path in glob('%s/*.pth' % opt.exp):
            os.remove(path)

        model_filename = '%s/epoch_%04d_dice%.4f_loss%.8f.pth' % (opt.exp, epoch+1, best_dice, losses.avg)

        # Single GPU
        if opt.ngpu == 1:
            torch.save(net.state_dict(), model_filename)
        # Multi GPU
        else:
            torch.save(net.module.state_dict(), model_filename)

    print('>>> Current best: Dice: %.8f in %3d epoch\n' % (best_dice, best_epoch+1))
    
    return best_dice, best_epoch


def evaluate(dataset_val, net, opt):
    print("Start Evaluation...")
    net.eval()

    for img, mask in tqdm(dataset_val):
        # Load Data
        img = torch.Tensor(img).float()
        if opt.use_gpu:
            img = img.cuda(non_blocking=True)

        # Predict
        with torch.no_grad():
            pred = net(img)

        raise NotImplementedError