
# CHECK THE NUMBER OF SKIPPED FRAMES IN THE VIDEOS

# The total number of skipped frames is 10, quite small

import torch
from torch.nn import functional as F
from torch import nn
from typing import Union
import numpy as np
import math
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
import glob

import feafa_utils
import feafa_dataloader
import feafa_architecture
import feafa_criterion


os.environ["CUDA_VISIBLE_DEVICES"]='2'

path = "/mmfs1/data/anzellos/data/FEAFA2"
window = 11
traindataset = feafa_dataloader.FeafaDataset(path,window,usage='Train')

videopaths = feafa_dataloader.findvideofolders(path)

# print(videopaths)
counter = 0
i = 0
for vp in videopaths:
    os.chdir(vp)
    path_frames = []
    expected_num = math.nan
    for frame_filename in sorted(glob.glob("*.jpg")):
        num = int(os.path.splitext(frame_filename)[0])
        if not math.isnan(expected_num): 
            if expected_num != num:
                print('Noncontiguous frames \n')
                print('Expected:',expected_num,'\n')
                print('Observed:',num,'\n')
                counter += 1
        expected_num = num+1
    i +=1
    print(i,' ')

print('Number of skipped frames:',counter)







