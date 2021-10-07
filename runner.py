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
import liteconfig
import mlflow
import mlflow.pytorch


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


from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr

from utils.loss import ComputeLoss
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, de_parallel
from utils.metrics import fitness


from get_model import GetModel
from dataset_loader import CreateDataset,CreateDataloader
from train import Train
#from models import load_model


RANK = int(os.getenv('RANK', -1))

def load_model(cfg):
    ML_FLOW_URL = cfg.Mlflow.mlflow_url
    mlflow.set_tracking_uri(ML_FLOW_URL)
    model_name = cfg.Mlflow.model_name
    model_version = cfg.Mlflow.model_version
    if os.path.isfile(f'{cfg.Detection.weights}.pt'):
        pass

    else:
        model = mlflow.pytorch.load_model(model_uri=f"models:/{model_name}/{model_version}")
        torch.save(model,f'{cfg.Detection.weights_savename}.pt')


    model = torch.hub.load(os.getcwd(), 'custom', path=f'{cfg.Detection.weights}.pt', source='local', force_reload = True)
    print('model loaded')
    return model

def run():
    print('Inside runner file')
    cfg = liteconfig.Config('config.ini')
    #check new data
    check_data = CreateDataset(cfg)
    if check_data.check_new_data():
        print('HERE')
        print('New data. Data will be added to the training folder')
        
        dataloader = CreateDataloader(cfg)
        trainloader,dataset = dataloader.get_dataloader()
        print('Data loaded Successfully')
        
        model = GetModel(cfg)
        model.load_model()
        print('Model Loaded')
        print(model.model)
        trainer = Train(cfg,trainloader=trainloader,model=model.model,RANK=RANK)
        trainer.begin_training()

    else:
        print("no new data found. Terminating the process")



if __name__ == '__main__':
    run()
