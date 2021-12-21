import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import math
import cv2
import random


label = []
for action in os.listdir('/home/jupyter-arangion/two-stream-pytorch-master/datasets/HAA500_frames/'):
    if action[0:-4] not in label:
        label.append(action[0:-4])


def findvideofolders(path):
    video_folders = []
    for action in os.listdir(path):
        action_path = os.path.join(path, action)
        video_folders.append((action_path, label.index(action[0:-4])))
    return video_folders


def windowmapper(videopaths,window):
    all_video = []
    all_labels = []
    for path, label in videopaths:
        os.chdir(path)
        path_frames = []
        #path_labels = []
        all_video.append(path)
        #for frame_filename in sorted(glob.glob("*.jpg")):
            #framepath = os.path.join(path,frame_filename)
            #path_frames.append(framepath)
        all_labels.append(label)
        
            
                   
    #print(len(all_video))
    return all_video, all_labels

                

class HAADataset(Dataset):
    def __init__(self, path, window, transform, usage, train=0.8):
        self.path = path
        self.window = window
        #self.transform = transforms.Compose([transforms.CenterCrop(720),transforms.Resize((224,224)),transforms.ToTensor()])
        self.transform = transform
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
        self.all_video, self.all_labels = windowmapper(self.videopaths,self.window) # THIS HAS BEEN MODIFIED FOR TESTING PURPOSES TO SHOW ONE VIDEO ONLY. CHANGE IT BACK IF YOU WANT TO TRAIN ON THE ENTIRE DATASET
        #print(self.allwindows_frames[0])
    def __getitem__(self, index):
        # read the frames in the window
        framepath = self.all_video[index]
        frames = []
        for frame_path in os.listdir(framepath):
            frame = Image.open(os.path.join(framepath, frame_path))
            frames.append(frame)
        #print(frames)
        input_frame = frames[random.randint(0, len(frames)-1)]
        if self.transform:
            input_frame = self.transform(input_frame)
        labels = self.all_labels[index]
        sample = {'frames':input_frame, 'label':torch.tensor(labels)}
        return sample

    def __len__(self):
        return len(self.all_video)