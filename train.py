import os
import shutil
import torch
import torch.nn as nn
from torch.optim import optim

from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr

from utils.loss import ComputeLoss
from torch.cuda import amp
import torch.optim.lr_scheduler as lr_scheduler



class Train():
    def __init__(self,cfg,trainloader,model,RANK):
        self.cfg = cfg
        self.trainloader = trainloader
        self.model = model
        self.nbs = 64
        self.optimizer = None
        self.hyp = None
        self.scheduler = None
        self.scaler = amp.Gradscaler(enabled=cuda)
        self.compute_loss = ComputeLoss(model)
        self.RANK = RANK
        self.device = self.cfg.Training.device


    def get_optimizer(self):
        if self.cfg.Training.adam:
            self.optimizer = optim.Adam(pg0, lr=self.hyp['lr0'], betas=(self.hyp['momentum'], 0.999))

        else:
            self.optimizer = optim.SGD(pg0, lr=self.hyp['lr0'], momentum=self.hyp['momentum'], nesterov=True)

    def get_scheduler(self):
        if self.cfg.Training.linear_lr:
            lf = lambda x: (1 - x / (self.cfg.Training.epochs - 1)) * (1.0 - self.hyp['lrf']) + self.hyp['lrf']  # linear

        else:
            lf = one_cycle(1, hyp['lrf'], self.cfg.Training.epochs)

        if self.optimizer:
            self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)

        else:
            print("Optimizer not defined.Cannot declare scheduler")




    def begin_training(self)->None:
        freeze = []
        for k,v in self.model.named_parameters():
            v.requires_grad = True
            if any(x in k for x in freeze):
                v.requires_grad=False

        accumulate = max(round(self.nbs/self.cfg.Training.batch_size))
        self.get_optimizer()
        self.get_scheduler()

        for epoch in range(self.start_epoch,self.cfg.Training.epochs):
            self.model.train()
            mloss = torch.zeros(4,device=self.device)
            if self.RANK != -1:
                self.trainloader.sampler.set_epoch(epoch)

            pbar = enumerate(self.trainloader)
            self.optimizer.zero_grad()
            for i,(imgs,targets,paths,_) in pbar:
                ni = i+nb*epoch
                imgs = imgs.to(self.device,non_blocking=True)

                #forward
                with amp.autocast(enabled=cuda):
                    pred = self.model(imgs)
                    loss,loss_item = self.compute_loss(pred,targets.to(self.device))
                    if self.cfg.Trainining.quad:
                        loss +=4.


                #backward
                self.scaler.scale(loss).backward()

                #optimizer

                if ni - last_opt_step >= accumulate:
                    self.scaler(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    last_opt_step = ni

                #print
                if self.RANK in [-1,0]:
                    mloss = (mloss*i+loss_item)/(i+1)
                    mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
                    s = ('%10s' * 2 + '%10.4g' * 6) % (
                            f'{epoch}/{self.cfg.Training.epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1])

                    pbar.set_description(s)

            self.scheduler.step()









