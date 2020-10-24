import torch
from PIL import Image
import gc
import os
import cv2
import glob
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms,utils,models
from torch.utils.data import Dataset,DataLoader

class dogDataset(Dataset):
    def __init__(self,path=None,size=256,transform=None):
        if path is None:
            path = '/media/yui/Disk/data/cat2dog/trainB/*.jpg'
        self.size = size
        self.transform = transform
        self.imgpaths = glob.glob(path)
    def __len__(self):
        return len(self.imgpaths)
    def __getitem__(self,idx):
        imgpath = self.imgpaths[idx]
        #img = cv2.imread(imgpath)[:,:,::-1]
        img = Image.open(imgpath)
        if self.transform:
            img = self.transform(img).cuda()
        return img/127.5 - 1

if __name__ == "__main__":
    trans = transforms.Compose([
            transforms.Resize((256,256)),transforms.ToTensor()])
    dog = dogDataset(transform=trans)
    if dog.transform is None:
        plt.imshow(dog[0]);plt.show()
    else:
        print(len(dog),dog[0].shape)

    gc.collect()
