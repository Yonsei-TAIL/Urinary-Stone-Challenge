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
from utils.losses import iou_modified

from matplotlib import pyplot as plt 


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


def evaluate(dataset_val, net, opt, save_dir):
    print("Start Evaluation...")
    net.eval()

    for idx, (img, mask) in enumerate(dataset_val):
        if idx%10 ==0:
            print("{}/{}".format(idx+1, len(dataset_val))) 
        # Load Data
        img = torch.Tensor(img).float()
        if opt.use_gpu:
            img = img.cuda(non_blocking=True)

        # Predict
        with torch.no_grad():
            pred = net(img)
	
            y = pred.sigmoid()
            dice = DiceCoef(return_score_per_channel=False)(y, mask.cuda())
            # Save original, image, label 
            # Convert to Binary
            zeros = torch.zeros(y.size())
            ones = torch.ones(y.size())
            y = y.cpu()

            y = torch.where(y > opt.threshold, ones, zeros) # threshold 0.99
            y = Variable(y).cuda()

            iou_score = iou_modified(y, mask.cuda(),opt)
            print(iou_score)
            print(iou_score.shape)

            ###### Plot & Save Figure #########
            origin = img.cpu().numpy()[0,0,:,:] 
            pred = y.cpu().numpy()[0,0,:,:]
            true = mask.cpu().numpy()[0,0,:,:]	

            fig = plt.figure()

            ax1 = fig.add_subplot(1,3,1)
            ax1.axis("off")
            ax1.imshow(origin, cmap = "gray")

            ax2= fig.add_subplot(1,3,2)
            ax2.axis("off")
            ax2.imshow(true, cmap = "gray")

            ax3 = fig.add_subplot(1,3,3)
            ax3.axis("off")
            ax3.imshow(origin,cmap = "gray")
            ax3.contour(pred, cmap='Reds', linewidths=0.5)

            plt.axis('off')
            plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)

            plt.savefig(opt.save_dir + "/original_label_pred_image_file_{}_dice_{}.png".format(idx, dice.item()),bbox_inces='tight', dpi=300)
            plt.cla()
            plt.close(fig)
            plt.gray()
            ###############################
