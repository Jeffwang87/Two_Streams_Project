import os
import time
import argparse
import shutil
import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import models
import datasets
import load_HAA_spatial_all
#import load_HAA_Sorted
import gc
#import GPUtil


#path = "/home/jupyter-arangion/two-stream-pytorch-master/datasets/HAA500_frames"
#window = 1


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
print(model_names)

dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', default='HAA500',
                    choices=["ucf101", "hmdb51", "HAA500"],
                    help='dataset: ucf101 | hmdb51 | HAA500')
parser.add_argument('--arch', '-a', metavar='ARCH', default='rgb_resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: reg_resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=250, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 50)')
parser.add_argument('--iter-size', default=2, type=int,
                    metavar='I', help='iter size as in Caffe to reduce memory usage (default: 5)')
parser.add_argument('--new_length', default=1, type=int,
                    metavar='N', help='length of sampled video frames (default: 1)')
parser.add_argument('--new_width', default=340, type=int,
                    metavar='N', help='resize width (default: 340)')
parser.add_argument('--new_height', default=256, type=int,
                    metavar='N', help='resize height (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[100, 200], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--save-freq', default=10, type=int,
                    metavar='N', help='save frequency (default: 25)')
parser.add_argument('--resume', default='/mmfs1/data/wangccy/Two_Streams_Project/spatial_final_save_2/model_best.pth.tar', type=str, 
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', default=0, type=int, metavar='N',
                    help='evaluate model on validation set N times')

best_prec1 = 0
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main():
    global args, best_prec1
    args = parser.parse_args()

    # create model
    print("Building model ... ")
    model = build_model()
    print("Model %s is loaded. " % (args.arch))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    if(args.resume[-8:] == '.pth.tar'):
        check = torch.load(args.resume)
        model.load_state_dict(check['state_dict'])
        optimizer.load_state_dict(check['optimizer'])
        start_epoch = check['epoch']
        best_prec1 = check['best_prec1']
        print('Testing model %s.' % (args.resume))
    else:
        if not os.path.exists(args.resume):
            os.makedirs(args.resume)
        start_epoch = args.start_epoch
        print("Saving everything to directory %s." % (args.resume))

    cudnn.benchmark = True

    # Data transforming
    scale_ratios = [1.0, 0.875, 0.75, 0.66]
    clip_mean = [0.485, 0.456, 0.406] * args.new_length
    clip_std = [0.229, 0.224, 0.225] * args.new_length

    normalize = transforms.Normalize(mean=clip_mean,
                                           std=clip_std)
    train_transform = transforms.Compose([
            transforms.Resize((256)),
            transforms.RandomResizedCrop((224, 224), (scale_ratios[1], scale_ratios[0]), (scale_ratios[3], scale_ratios[2])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    val_transform = transforms.Compose([
            transforms.Resize((256)),
            transforms.CenterCrop((224)),
            transforms.ToTensor(),
            normalize,
        ])
    
    #print(args.data)
    #print(sorted(os.listdir(args.data)))
    
    train_dataset = load_HAA.HAADataset(args.data, 
                                        args.new_length, 
                                        train_transform, 
                                        usage='Train')
    #print(train_dataset)
    
    val_dataset = load_HAA.HAADataset(args.data, 
                                      args.new_length, 
                                      val_transform, 
                                      usage='Test')
    #print(val_dataset)
    


    print('{} samples found'.format(len(train_dataset)))
    
    
    if args.evaluate > 0:
        val_avg_prec1, val_avg_prec3 = 0.0, 0.0
        
        for i in range(args.evaluate):
            val_loader = DataLoader(val_dataset, 
                                    batch_size = 32, 
                                    shuffle=True, 
                                    num_workers=args.workers, 
                                    pin_memory=True)
            
            val1, val3 = validate(val_loader, model, criterion)
            val_avg_prec1 += val1
            val_avg_prec3 += val3
            
        val_avg_prec1 /= args.evaluate
        val_avg_prec3 /= args.evaluate
        
        print('Averaged Validation Results: Prec@1 {top1:.3f} Prec@3 {top3:.3f}'
          .format(top1=val_avg_prec1, top3=val_avg_prec3))
        return
    
    
    for epoch in range(start_epoch, args.epochs):
        train_loader = DataLoader(train_dataset, 
                                  batch_size = 32, 
                                  shuffle=True, 
                                  num_workers=args.workers, 
                                  pin_memory=True)
        
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        #cudnn.benchmark = True
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = 0.0
        if (epoch + 1) % args.save_freq == 0:
            val_loader = DataLoader(val_dataset, 
                                    batch_size = 32, 
                                    shuffle=True, 
                                    num_workers=args.workers, 
                                    pin_memory=True)
            
            prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if (epoch + 1) % args.save_freq == 0:
            checkpoint_name = "%03d_%s" % (epoch + 1, "checkpoint.pth.tar")
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint_name, args.resume)

def build_model():

    model = models.__dict__['rgb_resnet18'](pretrained=True, num_classes=500)
    model.cuda()
    return model

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    optimizer.zero_grad()
    loss_mini_batch = 0.0
    acc_mini_batch = 0.0

    for i, data in enumerate(train_loader):
        #print(data)
        input = data['frames'].cuda(non_blocking=True)
        target = data['label'].cuda(non_blocking=True)
        #print(input.size())
        #print(target.size())
        #input = torch.reshape(input, (32, 3, 224, 224))
        #print(input.size())
        #print(target.size())
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
         
        #GPUtil.showUtilization()
        output = model(input_var)
        #GPUtil.showUtilization()
        del input_var
        
        # measure accuracy and record loss
        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        acc_mini_batch += float(prec1.item())
        loss = criterion(output, target_var)
        #GPUtil.showUtilization()
        del output, target_var
        loss = loss / args.iter_size
        loss_mini_batch += float(loss.data.item())
        loss.backward()
        #GPUtil.showUtilization()
        gc.collect()
        torch.cuda.empty_cache()
        #GPUtil.showUtilization()
        
        if (i+1) % args.iter_size == 0:
            # compute gradient and do SGD step
            optimizer.step()
            optimizer.zero_grad()

            # losses.update(loss_mini_batch/args.iter_size, input.size(0))
            # top1.update(acc_mini_batch/args.iter_size, input.size(0))
            losses.update(loss_mini_batch, input.size(0))
            top1.update(acc_mini_batch/args.iter_size, input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            loss_mini_batch = 0
            acc_mini_batch = 0
            
            if (i+1) % args.print_freq == 0:

                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       epoch, i+1, len(train_loader)+1, batch_time=batch_time, loss=losses, top1=top1))

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, data in enumerate(val_loader):
        
        
        input = data['frames'].cuda(non_blocking=True)
        target = data['label'].cuda(non_blocking=True)
        #print(input.size())
        #input = torch.reshape(input, (32, 3, 224, 224))
        #input_var = torch.tensor(input)
        #target_var = torch.tensor(target)
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)
        
        
        #del input_var
        # measure accuracy and record loss
        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        del output#, target_var
        gc.collect()
        torch.cuda.empty_cache()
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top3.update(prec3.item(), input.size(0))
        del input
        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top3=top3))
    
    if(args.evaluate > 0):
        return top1.avg, top3.avg
    else:
        print(' Validation Results: Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
          .format(top1=top1, top3=top3))
        return top1.avg

def save_checkpoint(state, is_best, filename, resume_path):
    cur_path = os.path.join(resume_path, filename)
    best_path = os.path.join(resume_path, 'model_best.pth.tar')
    torch.save(state, cur_path)
    if is_best:
        shutil.copyfile(cur_path, best_path)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 150 epochs"""

    decay = 0.1 ** (sum(epoch >= np.array(args.lr_steps)))
    lr = args.lr * decay
    print("Current learning rate is %4.6f:" % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
