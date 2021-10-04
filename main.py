import argparse
import logging
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, SGD, lr_scheduler
from tqdm import tqdm

from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    strip_optimizer, get_latest_run, check_dataset, check_git_status, check_img_size, check_requirements, \
    check_file, check_yaml, check_suffix, print_args, print_mutation, set_logging, one_cycle, colorstr, methods
from utils.downloads import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_labels, plot_evolve
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, intersect_dicts, select_device, \
    torch_distributed_zero_first
#from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.metrics import fitness
#from utils.loggers import Loggers
#from utils.callbacks import Callbacks

from datasets import ListDataset

#all the variables as per now

weights = 'yolo5s.pt' # initial weight path
cfg = 'model.yaml' #model.yaml path
data = 'data/coco128.yaml' #dataset.yaml path
hyp = 'hyp.yaml' # hyperparameters path


##TODO: create a class to return dataloader

#dataloader
RANK = int(os.getenv('RANK', -1))
save_dir = ''
# w = save_dir+'/weights'
# (w.parent if evolve else w).mkdir(parents=True,exist_ok=True)
# last, best = w/'last.pt', w/'best.wt'

init_seeds(1+RANK)
data_dict = 'dataset'
data = 'dataset'
with torch_distributed_zero_first(RANK):
    data_dict =  data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict, None

#train_loader,dataset = create_dataset()
def _create_data_loader(img_path,batch_size,img_size,n_cpu,multiscale_training=False):
    dataset = ListDataset(img_path,img_size=img_size,multiscale=multiscale_training,transform=None)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=n_cpu,pin_memory=True,collate_fn=dataset.collate_fn,worker_init_fn=worker_seed_set)
    return dataloader


#config
single_cls =True
nc = 1 if single_cls else int(data_dict['nc'])
names = ['plate'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']


#model
check_suffix(weights, '.pt')
pretrained = weights.endswith('.pt')
if pretrained:
    ckpt = torch.load(weights,map_location=device) #load checkpoint
    model = Model()  # load the model here



#train


#mlflow





