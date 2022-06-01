#Common imports
import os
import sys
import numpy as np
import argparse
import copy
import random
import json
import pickle

#Pytorch
import torch
from torch.autograd import grad
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.utils.data as data_utils

sys.path.append("./")
#robustdg
from utils.helper import *
from utils.match_function import *

from algorithms.erm import Erm

#gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# get args
parser = argparse.ArgumentParser()
args = parser.parse_args()
with open('configs/commandline_args.txt', 'r') as f:
    args.__dict__ = json.load(f)

if args.os_env:
    res_dir= os.getenv('PT_OUTPUT_DIR') + '/'
else:
    res_dir= 'results/'


base_res_dir=(
            res_dir + args.dataset_name + '/' + args.method_name + '/' + args.match_layer
            + '/' + 'train_' + str(args.train_domains)
        )
if not os.path.exists(base_res_dir):
    os.makedirs(base_res_dir)
#GPU
cuda= torch.device("cuda:" + str(args.cuda_device))
if cuda:
    kwargs = {'num_workers': 0, 'pin_memory': False}
else:
    kwargs= {}

#random constants
run = 0


#DataLoader
train_dataset = get_dataloader( args, run, args.train_domains, 'train', 0, kwargs )
val_dataset = get_dataloader(args, run, args.train_domains, 'val', 0, kwargs)
print(args.test_domains)
test_dataset = get_dataloader(args, run, args.test_domains, 'test', 0, kwargs)

print("defined network")

train_method = Erm(
    args, train_dataset, val_dataset,
    test_dataset, base_res_dir,
    run, cuda
)

# Train the method: It will save the model's weights post training and evalute it on test accuracy
train_method.train()


idx = np.argmax(train_method.val_acc)
target_acc = train_method.final_acc[idx]
src_acc = train_method.val_acc[idx]

print('Done for the Model..')
print(' Test Accuracy (Source Validation)', src_acc)
print('Test Accuracy (Target Validation)', target_acc)
print('\n')


print("Done training")
