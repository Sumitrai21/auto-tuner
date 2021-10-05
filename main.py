#import all the packages
#from dataset_loader import create_dataset
import argparse
import logging
import os
import random
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path
from threading import Thread

import math
import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

import val  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, de_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
from utils.metrics import fitness


#dump data to training folder
# src_path = '' #source path of the folder where data will be dumped
# dst_path = '' #path to the training data folder
# create_dataset(src_path,dst_path)

#load the data into the dataloader
train_pth = 'dataset/train'
imgsz = 312
batch_size=32
gs = 32
single_cls=True
hyp = 'data/hyps/hyp.scratch.yaml'
chache_images = False
rect = False
RANK = int(os.getenv('RANK', -1))
workers = 8
image_weights = False
quad = False
epochs = 2
device = 'cpu'

trainloader, dataset = create_dataloader(train_pth,imgsz,batch_size,gs,single_cls,hyp=hyp,
                        augment=True,cache=chache_images,rect=rect,rank=RANK,
                        workers=workers,image_weights=image_weights,quad=quad,prefix=colorstr('train: '))

#load the model
model = ''

#train the model
freeze = []
for k,v in model.named_parameters():
    v.requires_grad = True
    if any(x in k for x in freeze):
        v.requires_grad=False


nbs = 64
accumulate = max(round(nbs/batch_size),1)
hyp['weight_decay'] *= batch_size*accumulate/nbs

adam=True #take from config file

if adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
else:
    optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

if opt.linear_lr:
    lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
else:
    lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

#start training
t0 = time.time()
nw = max(round(hyp['warmup_epochs']*nb),1000)
last_opt_step = -1
maps = np.zeros(1)
results = (0,0,0,0,0,0,0)
start_epoch = 0
scheduler.last_epoch = start_epoch - 1
scaler = amp.Gradscaler(enabled=cuda)
compute_loss = ComputeLoss(model)

for epoch in range(start_epoch,epochs):
    model.train()
    if image_weights:
        if RANK in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
        if RANK != -1:
            indices = (torch.tensor(dataset.indices) if RANK == 0 else torch.zeros(dataset.n)).int()
            dist.broadcast(indices, 0)
            if RANK != 0:
                dataset.indices = indices.cpu().numpy()


    mloss = torch.zeros(4,device=device)
    if RANK != -1:
        trainloader.sampler.set_epoch(epoch)

    pbar = enumerate(trainloader)
    optimizer.zero_grad()
    for i,(imgs,targets,path,_) in pbar:
        ni = i+nb*epoch
        imgs = imgs.to(device,non_blocking=True)

        #forward
        with amp.autocast(enabled=cuda):
            pred = model(imgs)
            loss, loss_item = compute_loss(pred,targets.to(device))
            if RANK != -1:
                loss += WORLD_SIZE

            if quad:
                loss +=4.


        #backward
        scaler.scale(loss).backward()

        #optimize
        if ni- last_opt_step >= accumulate:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            last_opt_step = ni

        #print
        if RANK in [-1,0]:
            mloss = (mloss*i+loss_item)/(i+1) #it updates the mean loss
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 6) % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1])
            pbar.set_description(s)


    scheduler.step()




# save the results