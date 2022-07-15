import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SegmentationDataSet(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        data = np.load(self.data_path)
        self.imgs = data['x']
        self.masks = data['y']
        #for one hot encoding
        if not 'simple' in data_path:
            print('multi-class dataset. No additional class.')
            m = np.logical_not(self.masks)
            self.masks = np.concatenate([self.masks, m], axis = 1)
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        return [torch.as_tensor(self.imgs[idx], dtype = torch.float),
                torch.as_tensor(self.masks[idx], dtype = torch.float)]

        