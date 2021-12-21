import os
import time
import argparse
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import load_HAA_test
import tqdm
#import video_transforms
import models



def main():

    cudnn.benchmark = True
    
    path = "/home/wangccy/HAA500_frames"
    window = 11
    test_dataset = load_HAA_test.HAADataset(path,window,usage='Test')
    
    test_loader = torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size=32, shuffle=True,
                        num_workers=8, pin_memory=True)
    # create model
    print("Building model ...")
    temporal_model = build_temporal_model()
    spatial_model = build_spatial_model()
    
    temporal_model.eval()
    spatial_model.eval()
    #print("Model %s is loaded. " % (args.arch))
    total_currect_1 = 0
    total_currect_3 = 0
    for i, data in enumerate(tqdm.tqdm(test_loader)):
        input_rgb = data['frame'].cuda()
        input_flow = data['frames'].cuda()
        target = data['label'].cuda()
        with torch.no_grad(): 
            output_rgb = spatial_model(input_rgb) #[32, 500]
            output_flow = temporal_model(input_flow) #[32, 500]
        
        output_average = (output_rgb + output_flow)/2
        correct_1, correct_3 = accuracy(output_average.data, target, topk=(1,3))
        print(f"correct_1 is {correct_1:.3f} and correct_3 is {correct_3:.3f}")
        
        total_currect_1 += correct_1
        total_currect_3 += correct_3
        
        
        
    prec1 = total_currect_1 / len(test_dataset)
    prec3 = total_currect_3 / len(test_dataset)
    
    print(f"prec1 is {prec1:.3f} and prec3 is {prec3:.3f}")
        
        
        
  



def build_spatial_model():
    model = models.__dict__['rgb_resnet152'](pretrained=True, num_classes=500)
    check = torch.load("/home/wangccy/model_best.pth.tar")
    model.load_state_dict(check['state_dict'])
    model.cuda()
    return model
    

def build_temporal_model():
    model = models.flow_resnet.spatial_stream()
    check = torch.load("/home/wangccy/two-stream-pytorch-master/checkpoints/model_best.pth.tar")
    model.load_state_dict(check['state_dict'])
    model.cuda()
    return model


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k)
    return res


if __name__ == '__main__':
    main()
