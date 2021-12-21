import cv2
import os
import numpy as np
import time
import argparse
import torchvision.transforms as transforms

#transform = transforms.Compose([transforms.CenterCrop(720),
                                #transforms.Resize((224,224)),
                                #transforms.ToTensor()])

parser = argparse.ArgumentParser(description='Dataset Video Frame Converter')

parser.add_argument('src_dir', metavar='DIR',
                    help='path to dataset')
parser.add_argument('store_dir', metavar='DIR',
                    help='path to frame storage')

def FrameCapture(path, file, folder):
    #path = os.path.join(root, file)
    #file = file.replace(".mp4", "_")
    #folder = os.path.join(root, (file + "frames"))
    if(os.isdir(folder) == False):
      os.mkdir(folder)
    vid = cv2.VideoCapture(path)
    i=1

    while(vid.isOpened()):
        success, img = vid.read()
        if success == False:
            break
        #newImg = transforms.CenterCrop(720),transforms
        cv2.imwrite(os.path.join(folder, f"frame_{str(i-1).zfill(3)}.jpg"), img)
        
        i += 1
        
        
def VideoFinder(dataset, store):
    start = time.time()
    count = 0
    if os.isdir(store) == False:
      os.mkdir(store)
    for rt, dirs, fi in os.walk(dataset):
        for dir in dirs:
            for root, dr, files in os.walk(os.path.join(rt, dir)):
                for file in files:
                    if(".mp4" in file):
                        path = os.path.join(root, file)
                        name = file.replace(".mp4", "")
                        folder = os.path.join(store, name)
                        if(os.path.exists(os.path.join(folder, "frame_000.jpg"))):
                            break
                        FrameCapture(path, name, folder)
            print(dir, "completed!")
            count += 1
            print(f"{count}/500")
            print(f"Time to run: {int(time.time() - start)} seconds ")
        break

args = parser.parse_args()
VideoFinder(args.src_dir, args.store_dir)
