import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

path = "/mmfs1/data/anzellos/data/FEAFA2"

def findvideofolders(path):
    counter = 1
    for root, dirs, files in os.walk(path):
        if counter:
            firstleveldirs = dirs
            counter = 0
    
    fullpath3= []
    for path2 in firstleveldirs:
        fullpath2 = os.path.join(path,path2)
        counter = 1
        for root,dirs,files in os.walk(fullpath2):
            if counter:
                secondleveldirs = dirs
                counter = 0
            for path3 in secondleveldirs:
                fullpath3_temp = os.path.join(fullpath2,path3)
                if "_data" in fullpath3_temp:
                    fullpath3.append(fullpath3_temp)

    return fullpath3


def windowmapper(videopaths,window):
    allwindows_frames = []
    allwindows_labels = []
    for path in videopaths:
        os.chdir(path)
        path_frames = []
        path_labels = []

        for frame_filename in sorted(glob.glob("*.jpg")):
            framepath = os.path.join(path,frame_filename)
            path_frames.append(framepath)
            labelpath = framepath[:-4]+".auw"
            if os.path.isfile(labelpath):
                f = open(labelpath, "r")
                x = f.read()
                y = torch.from_numpy(np.asarray([float(i) for i in x.split()]))
                path_labels.append(y)
            else:
                path_labels.append("missing")

        for iwindow in range(len(path_frames)-window+1):
            missing = 0
            windowlabels = path_labels[iwindow:iwindow+window]
            for label in windowlabels:
                if type(label) == str:
                    missing+=1
            if not missing:
                framepaths = path_frames[iwindow:iwindow+window]
                allwindows_frames.append(framepaths)
                allwindows_labels.append(torch.cat(windowlabels,dim=0))
    
    return allwindows_frames, allwindows_labels


# path = "/mmfs1/data/anzellos/data/FEAFA2"
# window = 11
# videopaths = findvideofolders(path)
# videopaths = [videopaths[0]]
# allwindows_frames, allwindows_labels = windowmapper(videopaths,window)

class FeafaDataset(Dataset):
    def __init__(self, path, window):
        self.path = path
        self.window = window
        self.transform = transforms.ToTensor()
        # find all the frames
        self.videopaths = findvideofolders(self.path)
        # extract frames and labels
        self.allwindows_frames,self.allwindows_labels = windowmapper([self.videopaths[0]],self.window)

    def __getitem__(self, index):
        # read the frames in the window
        framepaths = self.allwindows_frames[index]
        frames = []
        for frame_path in framepaths:
            frame = Image.open(frame_path)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        x = torch.stack(frames,dim=1)
        labels = self.allwindows_labels[index]
        sample = {'frames':x,'labels':labels}
        return sample

    def __len__(self):
        return len(self.allwindows_frames)

# path = "/mmfs1/data/anzellos/data/FEAFA2"
# window = 11
# trainingset = FeafaDataset(path,window)
# 
# print(trainingset.__len__(),'\n')
# data = trainingset.__getitem__(0)
# print(data['frames'],'\n')
# print(data['labels'],'\n')



