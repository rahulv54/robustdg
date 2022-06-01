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

#tent imports
import logging
import tent.tent as tent
from tent.conf import cfg, load_cfg_fom_args, load_cfg

#robustDG
from algorithms.erm import Erm
from utils.helper import *
from utils.match_function import *

#GPU
cuda= torch.device("cuda:" + str(0))
if cuda:
    kwargs = {'num_workers': 0, 'pin_memory': False}
else:
    kwargs= {}


logger = logging.getLogger(__name__)
fh = logging.FileHandler('logs/spam.log')
logger.addHandler(fh)

def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError


def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model


def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model_slab(model)
    params, param_names = tent.collect_params_slab(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    # logger.info(f"model for adaptation: %s", model)
    # logger.info(f"params for adaptation: %s", param_names)
    # logger.info(f"optimizer for adaptation: %s", optimizer)

    print(f"model for adaptation: %s", model)
    print(f"params for adaptation: %s", param_names)
    print(f"optimizer for adaptation: %s", optimizer)
    return tent_model

def get_accuracy(model, dataset):
    test_acc, test_size = 0, 0
    data_loader = dataset['data_loader']
    for batch_idx, (x_e, y_e, d_e, idx_e, obj_e) in enumerate(data_loader):
        x_e = x_e.to(cuda)
        y_e = torch.argmax(y_e, dim=1).to(cuda)

        # Forward Pass
        out = model(x_e)

        test_acc += torch.sum(torch.argmax(out, dim=1) == y_e).item()
        test_size += y_e.shape[0]

    return test_acc / test_size

def evaluate(train_dataset, val_dataset, test_dataset, args):

    load_cfg("tent/cfgs/", "tent.yaml")
    # configure model


    if args.os_env:
        res_dir = os.getenv('PT_OUTPUT_DIR') + '/'
    else:
        res_dir = 'results/'

    base_res_dir = (
            res_dir + args.dataset_name + '/' + args.method_name + '/' + args.match_layer
            + '/' + 'train_' + str(args.train_domains)
    )
    if not os.path.exists(base_res_dir):
        os.makedirs(base_res_dir)

    # random constants
    run = 0

    train_method = Erm(
        args, train_dataset, val_dataset,
        test_dataset, base_res_dir,
        run, cuda
    )

    model_name = 'Model_' + train_method.post_string + '.pth'
    print("Output Folder:", os.path.join(base_res_dir, model_name))

    base_model = train_method.phi
    base_model.load_state_dict(torch.load(os.path.join(base_res_dir, 'Model_' +
                                                       train_method.post_string + '.pth')))



    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model)

    if cfg.MODEL.ADAPTATION == "tent":
        logger.info("test-time adaptation: TENT")
        model = setup_tent(base_model)

    err = get_accuracy(model, test_dataset)
    print(f"accuracy %  {err:.2%}")

if __name__ == '__main__':
    # get args
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open('configs/commandline_args.txt', 'r') as f:
        args.__dict__ = json.load(f)

    run=0

    # DataLoader
    train_dataset = get_dataloader(args, run, args.train_domains, 'train', 0, kwargs)
    val_dataset = get_dataloader(args, run, args.train_domains, 'val', 0, kwargs)
    print(args.test_domains)
    test_dataset = get_dataloader(args, run, args.test_domains, 'test', 0, kwargs)

    evaluate(train_dataset, val_dataset, test_dataset, args)