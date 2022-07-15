
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from unet.evaluate import evaluate
from segmentation_experiments.data_loading import SegmentationDataSet
from segmentation_experiments import data_loading
from utils.dice_score import dice_loss
from unet import UNet
from unet import simpleUNet



# In[3]:


from importlib import reload
reload(data_loading)
reload(simpleUNet)


# In[4]:

train_sets = glob.glob('data/syntheticSegmentation/simple_train_dom1_*')
val_set_name = glob.glob('data/syntheticSegmentation/simple_balanced_*')[0]

for train_set_name in train_sets[2:]:

    train_set = data_loading.SegmentationDataSet(train_set_name)
    val_set = data_loading.SegmentationDataSet(val_set_name)
    batch_size = 32
    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
    }


    # In[5]:


    #check outputs from dataloader

    inputs, masks = next(iter(dataloaders['train']))
    print(inputs.shape, masks.shape)


    # In[6]:


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = simpleUNet.UNet(n_class=2)
    model = model.to(device)

    # check keras-like model summary using torchsummary
    from torchsummary import summary
    summary(model, input_size=(1, 256, 256))


    # In[ ]:


    import torch
    import torch.optim as optim
    from torch.optim import lr_scheduler
    from unet import training_loop
    reload(training_loop)

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
     
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    model_name = os.path.basename(train_set_name)[:-len('.npz')]
    model, loss_values = training_loop.train_model(model, optimizer_ft, exp_lr_scheduler, dataloaders, model_name, num_epochs=85)

    np.save('checkpoints/' + model_name + '_Metrics.npz', loss_values)



