import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import math

label = []
for action in os.listdir('/home/wangccy/HAA500_frames/'):
    if action[0:-4] not in label:
        label.append(action[0:-4])


def findvideofolders(path):
    video_folders = []
    for action in os.listdir(path):
        action_path = os.path.join(path, action)
        video_folders.append((action_path, label.index(action[0:-4])))
    return video_folders


def windowmapper(videopaths,window):
    allwindows_frames = []
    allwindows_labels = []
    for path, label in videopaths:
        os.chdir(path)
        path_frames = []
        #path_labels = []
            
        for frame_filename in sorted(glob.glob("*.jpg")):
            framepath = os.path.join(path,frame_filename)
            path_frames.append(framepath)
            
                    
        for iwindow in range(len(path_frames)-window+1):
            framepaths = path_frames[iwindow:iwindow+window]
            allwindows_frames.append(framepaths)
            allwindows_labels.append(label)
    return allwindows_frames, allwindows_labels

                
        
        

class HAADataset(Dataset):
    def __init__(self, path, window,usage,train=0.8):
        self.path = path
        self.window = window
        self.transform = transforms.Compose([transforms.CenterCrop(720),transforms.Resize((224,224)),transforms.ToTensor()])
        # find all the frames
        videopaths = findvideofolders(self.path)
        # select train or test
        if usage == 'Train':
            self.videopaths = videopaths[:math.ceil(len(videopaths)*train)]
        elif usage == 'Test':
            self.videopaths = videopaths[math.ceil(len(videopaths)*train):]
        else:
            print('Error: incorrect usage')
        # extract frames and labels
        self.allwindows_frames, self.allwindows_labels = windowmapper(self.videopaths,self.window) # THIS HAS BEEN MODIFIED FOR TESTING PURPOSES TO SHOW ONE VIDEO ONLY. CHANGE IT BACK IF YOU WANT TO TRAIN ON THE ENTIRE DATASET
       # print(self.allwindows_frames[0])
    def __getitem__(self, index):
        # read the frames in the window
        framepaths = self.allwindows_frames[index]
        frames = []
        for frame_path in framepaths:
            frame_temp = Image.open(frame_path)
            if self.transform:
                frame_temp = self.transform(frame_temp)
            frames.append(frame_temp)
        frame = frames[0]
        x = torch.stack(frames,dim=1)
        labels = self.allwindows_labels[index]
        
        sample = {'frame':frame, 'frames':x, 'label':torch.tensor(labels)}
        return sample

    def __len__(self):
        return len(self.allwindows_frames)
