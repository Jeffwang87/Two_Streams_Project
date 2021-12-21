
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

savepath = '/mmfs1/data/anzellos/results/twostream_feafa'
for i in range(5):
    filename = 'epoch'+str(i+1).zfill(5)+'.ckp'
    filepath = os.path.join(savepath,filename)
    # print(filepath,'\n')
    checkpoint = torch.load(filepath,map_location='cpu')
    loss = checkpoint['loss']
    print(loss,'\n')
