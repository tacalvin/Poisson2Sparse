import os

from PIL import Image

import torch
import torchvision
from torchvision import transforms
import numpy as np
from torchvision.transforms.transforms import CenterCrop

from glob import glob


class FMDDataset():
    def __init__(self, path):
        self.path = path
        self.gt_root = os.path.join(path, 'gt/')
        self.raw_root = os.path.join(path, 'raw/')
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

        self.data = []
        
        # 20 elements
        gt_elements = glob(self.gt_root+'*')
        # print(self.gt_root + "*")
        # print(gt_elements)
        gt_elements.sort()
        # print(gt_elements)
        raw_elements = glob(self.raw_root + '*')
        raw_elements.sort()
        for i in range(len(gt_elements)):
            self.data.append( {
                'gt': gt_elements[i],
                'raw': raw_elements[i]
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # img_files = glob(self.data[index]['raw']+'*png')
        img_files = self.data[index]['raw']
        # print(self.data[index]['raw']+'*png')
        # img_files.sort()
        img =  self.transforms(Image.open(img_files))
        gt = self.transforms(Image.open(self.data[index]['gt'] ))

        # print(img, torch.max(img), torch.min(img))
        # quit()
        return img, gt
