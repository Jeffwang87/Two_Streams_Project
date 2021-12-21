
savepath = '/mmfs1/data/anzellos/results/twostream_feafa/epoch00002.ckp'


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
import feafa_architecture
import feafa_criterion


os.environ["CUDA_VISIBLE_DEVICES"]='2'

path = "/mmfs1/data/anzellos/data/FEAFA2"
window = 11
traindataset = feafa_dataloader.FeafaDataset(path,window,usage='Train')
trainloader = DataLoader(traindataset,batch_size = 32)
 
flownet = feafa_architecture.TinyMotionNet()
flownet.cuda()
flownet.train()

optimizer = optim.SGD(flownet.parameters(), lr=0.1, momentum=0.9)

checkpoint = torch.load(savepath)
flownet.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

data = traindataset.__getitem__(0)
frames = data['frames'].unsqueeze(0).cuda()
flows = flownet(frames)
print(loss,'\n')
print(flows[0].shape,'\n')
