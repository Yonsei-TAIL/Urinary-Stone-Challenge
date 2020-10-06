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
from utils.losses import iou_modified, avg_precision

from matplotlib import pyplot as plt 


def train(net, dataset_trn, optimizer, criterion, epoch, opt):
    print("Start Training...")
    net.train()

    losses, total_dices, total_iou = AverageMeter(), AverageMeter(), AverageMeter()

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

        pred = pred.sigmoid()
        # Backward and step
        loss.backward()
        optimizer.step()
        
        # Calculation Dice Coef Score
        dice = DiceCoef(return_score_per_channel=False)(pred, mask)
        total_dices.update(dice.item(), img.size(0))
        
        # Convert to Binary
        zeros = torch.zeros(pred.size())
        ones = torch.ones(pred.size())
        pred = pred.cpu()

        pred = torch.where(pred > 0.5, ones, zeros).cuda() # threshold 0.99

        # Calculation IoU Score
        iou_score = iou_modified(pred, mask,opt)

        total_iou.update(iou_score.mean().item(), img.size(0))

        # Stack Results
        losses.update(loss.item(), img.size(0))

        if (it==0) or (it+1) % 10 == 0:
            print('Epoch[%3d/%3d] | Iter[%3d/%3d] | Loss %.4f | Dice %.4f | Iou %.4f'
                % (epoch+1, opt.max_epoch, it+1, len(dataset_trn), losses.avg, total_dices.avg, total_iou.avg))

    print(">>> Epoch[%3d/%3d] | Training Loss : %.4f | Dice %.4f | Iou %.4f\n "
        % (epoch+1, opt.max_epoch, losses.avg, total_dices.avg, total_iou.avg))


def validate(dataset_val, net, criterion, epoch, opt, best_iou, best_epoch):
    print("Start Evaluation...")
    net.eval()

    # Result containers
    losses, total_dices, total_iou = AverageMeter(), AverageMeter(), AverageMeter()

    for it, (img, mask) in enumerate(dataset_val):
        # Load Data
        img, mask = torch.Tensor(img).float(), torch.Tensor(mask).float()
        if opt.use_gpu:
            img, mask = img.cuda(non_blocking=True), mask.cuda(non_blocking=True)

        # Predict
        pred = net(img)

        # Loss Calculation
        loss = criterion(pred, mask)

        pred = pred.sigmoid()

        # Calculation Dice Coef Score
        dice = DiceCoef(return_score_per_channel=False)(pred, mask)
        total_dices.update(dice.item(), img.size(0))
        
        # Convert to Binary
        zeros = torch.zeros(pred.size())
        ones = torch.ones(pred.size())
        pred = pred.cpu()

        pred = torch.where(pred > 0.5, ones, zeros).cuda()
        
        # Calculation IoU Score
        iou_score = iou_modified(pred, mask,opt)

        total_iou.update(iou_score.mean().item(), img.size(0))

        # Stack Results
        losses.update(loss.item(), img.size(0))

        # if (it==0) or (it+1) % 10 == 0:
        #     print('Epoch[%3d/%3d] | Iter[%3d/%3d] | Loss %.4f | Dice %.4f | Iou %.4f'
        #         % (epoch+1, opt.max_epoch, it+1, len(dataset_trn), losses.avg, total_dices.avg, total_iou.avg))

    print(">>> Epoch[%3d/%3d] | Test Loss : %.4f | Dice %.4f | Iou %.4f"
        % (epoch+1, opt.max_epoch, losses.avg, total_dices.avg, total_iou.avg))

    # Update Result
    if total_iou.avg > best_iou:
        print('Best Score Updated...')
        best_iou = total_iou.avg
        best_epoch = epoch

        # # Remove previous weights pth files
        # for path in glob('%s/*.pth' % opt.exp):
        #     os.remove(path)

        model_filename = '%s/epoch_%04d_iou%.4f_loss%.8f.pth' % (opt.exp, epoch+1, best_iou, losses.avg)

        # Single GPU
        if opt.ngpu == 1:
            torch.save(net.state_dict(), model_filename)
        # Multi GPU
        else:
            torch.save(net.module.state_dict(), model_filename)

    print('>>> Current best: IoU: %.8f in %3d epoch\n' % (best_iou, best_epoch+1))
    
    return best_iou, best_epoch


def evaluate(dataset_val, net, opt, save_dir):
    print("Start Evaluation...")
    net.eval()

    iou_scores = []
    for idx, (img, mask) in enumerate(dataset_val):
        # Load Data
        img = torch.Tensor(img).float()
        if opt.use_gpu:
            img = img.cuda(non_blocking=True)

        # Predict
        with torch.no_grad():
            pred = net(img)
	
            y = pred.sigmoid()
            dice = DiceCoef(return_score_per_channel=False)(y, mask.cuda())
            
            # Convert to Binary
            zeros = torch.zeros(y.size())
            ones = torch.ones(y.size())
            y = y.cpu()

            y = torch.where(y > opt.threshold, ones, zeros) # threshold 0.99
            y = Variable(y).cuda()

            iou_score = iou_modified(y, mask.cuda(),opt)

            if idx%10 ==0:
                print("{}/{} - dice {} | IoU {}".format(idx+1, len(dataset_val), dice.item(), iou_score.item()))

            iou_scores.append(iou_score.item())

            if iou_score < 0.75:
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
                ax2.imshow(origin,cmap = "gray")
                ax2.contour(true, cmap='Greens', linewidths=0.1)

                ax3 = fig.add_subplot(1,3,3)
                ax3.axis("off")
                ax3.imshow(origin,cmap = "gray")
                ax3.contour(pred, cmap='Reds', linewidths=0.5)

                plt.axis('off')
                plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)

                plt.savefig(opt.save_dir + "/original_label_pred_image_file_{}_dice_{}_iou_{}.png".format(idx, dice.item(),iou_score.item()),bbox_inces='tight', dpi=300)
                plt.cla()
                plt.close(fig)
                plt.gray()
                ###############################

    prec_thresh1, prec_thresh2, iou_mean = avg_precision(iou_scores)

    print("Average Presion with threshold 0.5 {}, 0.75 {}, Mean {}".format(prec_thresh1, prec_thresh2, iou_mean))
