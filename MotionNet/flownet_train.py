import torch
from torch.nn import functional as F
from torch import nn
from typing import Union
import numpy as np
from math import exp
import os
from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision import models
import torch.optim as optim
import gc

import feafa_utils
import feafa_dataloader
import load_HAA
import feafa_architecture
import feafa_criterion


os.environ["CUDA_VISIBLE_DEVICES"]='2'

path = "/home/wangccy/HAA500_frames"
window = 11
traindataset = load_HAA.HAADataset(path,window,usage='Train')
trainloader = DataLoader(traindataset,batch_size = 32, num_workers = 8, pin_memory=True)
 
flownet = feafa_architecture.TinyMotionNet()
flownet.cuda()
#flownet.to(torch.device('cuda'))
#flownet = nn.DataParallel(flownet, device_ids=[0,1,2,3])
flownet.train()

reconstructor = feafa_utils.Reconstructor()
#reconstructor.to(torch.device('cuda'))
#reconstructor = nn.DataParallel(reconstructor, device_ids=[0,1,2,3])

criterion = feafa_criterion.SimpleLoss(flownet)

optimizer = optim.SGD(flownet.parameters(), lr=0.01, momentum=0.9)

save_root = "/home/wangccy/twostream_feafa/saved2"

save_freq = 1                               # specify every how many epochs to save the model
loss_memory = []
for epoch in range(10):  # loop over the dataset multiple times
    print('Starting epoch ',epoch,' ...\n')
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # get the inputs
        print(f"epoch is {epoch}, progress is {i+1}/{len(trainloader)+1}")
        frames = data['frames'].cuda()
        torch.cuda.empty_cache()
        flows = flownet(frames)
        #flows.to(torch.device('cuda'))
        t0s, reconstructed, flows_reshaped = reconstructor(frames, flows) # t0s are original images excluding the 11th, downsampled to match the reconstructed versions
        # zero the parameter gradients
        frames.detach().cpu()
        for flow in flows:
            flow.detach().cpu()
        del flows,frames
        gc.collect()
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        # forward + backward + optimize
        loss = criterion(t0s,reconstructed,flows_reshaped,flownet)
        for t0 in t0s:
            t0.detach().cpu()
        for reco in reconstructed:
            reco.detach().cpu()
        for flore in flows_reshaped:
            flore.detach().cpu()
        del t0s,reconstructed,flows_reshaped
        gc.collect()
        torch.cuda.empty_cache()
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.data.item()
    epoch_loss = running_loss / len(trainloader)
    print('[%d] loss: %.3f' %(epoch + 1, epoch_loss ))
    f = open("/home/wangccy/twostream_feafa/saved2/result.txt", "a")
    f.write('[%d] loss: %.3f' %(epoch + 1, epoch_loss ))
    f.close()

    # loss_memory.append(epoch_loss)
    running_loss = 0.0
    if epoch % save_freq == save_freq-1: 
        savename = f'epoch{epoch+1:05d}.ckp'
        save_path = os.path.join(save_root,savename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': flownet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss
            }, save_path)


