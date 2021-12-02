#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os, sys, gc
import time
import glob
import pickle
import copy
import json
import random
from collections import OrderedDict, namedtuple
import multiprocessing
import threading
import traceback

from typing import Tuple, List

import h5py
from tqdm import tqdm, tqdm_notebook

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
from PIL import Image


import torch
import torchvision
import torch.nn.functional as F

from torch import nn, optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchmetrics
import pl_bolts
import pytorch_lightning as pl

import timm
import timm.loss


# In[6]:


class Args():
    # Senttng default parameters
    
    def get_argslist(self):
        arg_list = [p for p in  dir(self) if (not p.startswith('_')) and (not p.startswith('get'))]
        return arg_list
    
    def get_argsdict(self):
        arg_list = self.get_argslist()
        arg_dict = {k : getattr(self, k) for k in arg_list}
        return arg_dict
    
    def __repr__(self):
        s = 10*' =' + ' ArgList' + 10*' =' + '\n'
        for k, v in self.get_argsdict().items():
            s += f'{k:30s}: {v}\n'
            
        return s

    def __str__(self):
        return self.__repr__()
    
    
class ArgsT22_EffNetV2(Args):
    OUTPUT_WH            = (160, 160)
    DATASET_WH           = (384, 384)
    BATCH_SIZE           = 96
    N_WORKERS            = 28
    BACKBONE_GPUS        = [0]
    DS_INPUT_DIR         = './all_datasets/dataset'
    DS_DIR               = f'{DS_INPUT_DIR}_jpg_{DATASET_WH[0]}x{DATASET_WH[1]}' # Path where the rescaled images will be saved
    ALL_FOLDERS          = ['query_images', 'reference_images', 'training_images', 'imagenet_images', 'face_frames']
    
    ImgNet_SAMPLES       = './ImageNet_samples_v.pickle'
    FrmFaces_SAMPLES     = './FrameFaces_samples_v.pickle'
    
    img_norm_type        = ['simple', 'imagenet', 'centered'][1]
    model_resolution     = (160, 160)
    embedding_size       = 2*128
    backbone_name        = ['efficientnetv2_rw_s', 'efficientnetv2_rw_m', 'tf_efficientnetv2_l_in21ft1k'][0]
    pretrained_bb        = True
    neck_type            = 'A'
    eps                  = 1e-6
    init_lr              = 1e-3
    use_scheduler        = True
    scheduler_name       = ['ReduceLROnPlateau', 'LinearWarmupCosineAnnealingLR'][1]
    
    sched_factor         = 0.5
    sched_patience       = 3
    
    n_warmup_epochs      = 4
    n_decay_epochs       = 100                           
    lr_prop              = 1e-3
    
    
    optimizer_name       = ['adam', 'sgd'][0]
    clip_grad_norm       = 1.0
    weight_decay         = 0.0
    n_steps_grad_update  = 1
    start_ckpt           = None
    criterion_name       = ['arcface', 'arcface_multigpu'][1]
    
    
    do_ref_trn           = False
    ref_trn_n_epochs     = 5
    
    n_facebook_samples   = 1_000_000  # 1_000_000 Facebook's training dataset
    n_imagenet_samples   = 1_000_000  # 1_431_167 ImageNet
    n_frm_face_samples   = 0          #   400_000  #   476_584 FrameFaces (Google FeepFake's dataset)
    
    arc_classnum         = n_facebook_samples + n_imagenet_samples + n_frm_face_samples
    arc_devices_v        = [0] * 30 + [0] * 10
    arc_w_prop_v         = None
    arc_optimizer        = None # NotUsed
    
    arc_m                = 0.4
    arc_s                = 40.0
    arc_bottleneck       = None
    arc_weight_decay     = 0.0
    use_GeM              = True
    GeM_p                = 3.0
    GeM_opt_p            = True
    checkpoint_base_path = './TEST23_MiltiGPU_AF'
    model_name           = 'arc_model'
    
    precision            = [32, 16][1]
    seed                 = None
    
    accelerator          = ['dp', 'ddp'][1]
    
    
# args = ArgsT22_EffNetV2()
# print(args)


# # Model

# In[11]:


def l2_norm(x, axis=1, eps=1e-6):
    norm = torch.norm(x, 2, axis, True) + eps
    output = torch.div(x, norm)
    return output


class FocalLoss(nn.Module):
    def __init__(self, gamma=3.0, positve_weighting=-1):
        super().__init__()
        self.alpha = positve_weighting
        self.gamma = gamma
        
        return None

    def forward(self, inputs, targets, reduction='mean'):
        focal_loss = torchvision.ops.sigmoid_focal_loss(
            inputs,
            targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=reduction,
        )
        
        return focal_loss


# In[12]:


class ArcfaceMultiGPU(nn.Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599    
    def __init__(self, args):
        super().__init__()
        
        self.args           = args
        self.classnum       = args.arc_classnum
        self.embedding_size = args.embedding_size
        self.s              = args.arc_s  # the margin value, default is 0.5
        self.m              = args.arc_m  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        
        self.optimizer_name = args.arc_optimizer
        self.scheduler_name = args.scheduler_name
        
        self.devices_v      = args.arc_devices_v
        self.w_porp_v       = args.arc_w_prop_v
        self.init_lr        = args.init_lr
        self.weight_decay   = args.weight_decay
        self.lr_prop        = args.lr_prop
        self.sched_factor   = args.sched_factor
        self.sched_patience = args.sched_patience
        self.n_warmup_epochs= args.n_warmup_epochs
        self.n_decay_epochs = args.n_decay_epochs
        self.scheduler_name = args.scheduler_name
        self.use_scheduler  = args.use_scheduler
        
        self.devices_v = [torch.device(d) for d in self.devices_v]
        self.n_devs = len(self.devices_v)
        
        if self.w_porp_v is None:
            self.w_porp_v = [1/self.n_devs for i in range(self.n_devs)]
            
        porp_s = sum( self.w_porp_v )
        self.w_porp_v = [p/porp_s for p in self.w_porp_v]
        
        self.n_cls_v = []
        for i_p, p in enumerate(self.w_porp_v):
            if i_p != self.n_devs - 1:
                self.n_cls_v.append(
                    int( round(self.classnum * p) )
                )
                
            else:
                self.n_cls_v.append(
                    self.classnum - sum(self.n_cls_v) 
                )

        self.kernel_v = []
        for n_cls, device in zip(self.n_cls_v, self.devices_v):
            
            kernel = torch.Tensor(self.embedding_size, n_cls).uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
            kernel = kernel.to(device)
            kernel = nn.Parameter(
                kernel
            )
            
            self.kernel_v.append( kernel )
        
        self.kernel_v = nn.ParameterList( self.kernel_v )

        
        self.cos_m = np.cos(self.m)
        self.sin_m = np.sin(self.m)
        
        self.mm = (self.sin_m * self.m) # issue 1
        self.threshold = np.cos(np.pi - self.m)
        
        
        self.criterion = nn.CrossEntropyLoss(
                weight=None,
                reduction='mean',
            )
        
#         self._set_devices()
        
        return None
    
    
    def _set_devices(self, opt=None): 
        print(' - ArcfaceMultiGPU, setting devices:')
        for i_k, device in enumerate(self.devices_v):
            self.kernel_v[i_k].data = self.kernel_v[i_k].data.to(device)
            print(f'  |-> kernel({i_k:2d}): {self.kernel_v[i_k].device}   shape: {self.kernel_v[i_k].shape}')
            
        if opt is not None:
            pass
        return None
        
    
#     def project(self, embbedings):
#         # weights norm
#         nB = len(embbedings)
        
#         kernel_norm = l2_norm(self.kernel, axis=0)
        
#         # cos(theta+m)
#         cos_theta = torch.mm(embbedings, kernel_norm)
#         cos_theta = cos_theta.clamp(-1,1) # for numerical stability
#         output = self.s * cos_theta
#         return output
        
    
        
        
    def forward(self, embbedings, label):

        def _worker(
            i_th,
            kernel,
            embbedings,
            results_d,
            cos_m, sin_m, mm, threshold,
            device,
            th_lock,
            grad_enabled,
            autocast_enabled,
        ):
            try:
                torch.set_grad_enabled(grad_enabled)

    #             print('TH:', i_th, embbedings.shape, kernel.shape, kernel.device)

                with torch.cuda.device(device), torch.cuda.amp.autocast(enabled=autocast_enabled):
                    embbedings = embbedings.to(device)

                    kernel_norm = l2_norm(kernel, axis=0)

                    # cos(theta+m)

                    cos_theta = torch.mm(embbedings, kernel_norm)
                    cos_theta = cos_theta.clamp(-1,1) # for numerical stability

                    cos_theta_2 = torch.pow(cos_theta, 2).type(cos_theta.dtype)
                    sin_theta_2 = 1.0 - cos_theta_2
                    sin_theta = torch.sqrt(sin_theta_2)
                    cos_theta_m = (cos_theta * cos_m - sin_theta * sin_m)

                    # this condition controls the theta+m should in range [0, pi]
                    #  0<=theta+m<=pi
                    # -m<=theta<=pi-m
                    cond_v = cos_theta - threshold
                    cond_mask = cond_v <= 0

                    keep_val = (cos_theta - mm) # when theta not in [0, pi], use cosface instead

                    cos_theta_m[cond_mask] = keep_val[cond_mask]
                    output = cos_theta * 1.0 # a little bit hacky way to prevent in_place operation on cos_theta


                with th_lock:
                    results_d[i_th] = (output, cos_theta, cos_theta_m)
                    
            except Exception as e:
                with th_lock:
                    print(' -'*40 + f'\n ERROR i_th={i_th}: ', e, file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                
                raise e
                
            return None
        
        
        th_lock = threading.Lock()
        results_d = {}
        grad_enabled, autocast_enabled = torch.is_grad_enabled(), torch.is_autocast_enabled()
        
        th_v = []
        i_cls = 0
        for i_th, device in enumerate(self.devices_v):
            th = threading.Thread(
                target=_worker,
                args=(
                    i_th,
                    self.kernel_v[i_th],
                    embbedings,
                    results_d,
                    self.cos_m, self.sin_m, self.mm, self.threshold,
                    device,
                    th_lock,
                    grad_enabled,
                    autocast_enabled
                )
            )
            
            th_v.append( th )

        for th in th_v:
            th.start()
            
        for th in th_v:
            th.join()
        
        
        embbed_device = embbedings.device
        
        
        output_v, cos_theta_v, cos_theta_m_v = [], [], []
        for i_th, device in enumerate(self.devices_v):
            output, cos_theta, cos_theta_m = results_d[i_th]
            
            output = output.to(embbed_device)
            cos_theta = cos_theta.to(embbed_device)
            cos_theta_m = cos_theta_m.to(embbed_device)
            
            
            output_v.append(output)
            cos_theta_v.append(cos_theta)
            cos_theta_m_v.append(cos_theta_m)
        
        # Concatenating results...
        output = torch.cat(output_v, dim=1)
        cos_theta = torch.cat(cos_theta_v, dim=1)
        cos_theta_m = torch.cat(cos_theta_m_v, dim=1)
        
        nB = len(embbedings)
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s # scale up in order to make softmax work, first introduced in normface
        
        loss = self.criterion(
            output,
            label,
        )
        
        return loss, output, cos_theta
    
    @torch.jit.ignore
    def _build_scheduler(self, LAST_EPOCH=None):
        if self.scheduler_name.lower() == 'LinearWarmupCosineAnnealingLR'.lower():
            self.scheduler = pl_bolts.optimizers.LinearWarmupCosineAnnealingLR(
                optimizer=self.optimizer,                                                               
                warmup_epochs=self.n_warmup_epochs,
                max_epochs=self.n_decay_epochs+self.n_warmup_epochs,
                warmup_start_lr=10 * self.lr_prop * self.init_lr,
                eta_min=self.lr_prop * self.init_lr,
                last_epoch=-1,
            )
            
            scheduler_d = {
                "scheduler": self.scheduler,
            }
            
        elif self.scheduler_name.lower() == 'ReduceLROnPlateau'.lower():
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.sched_factor,
                patience=self.sched_patience,
                verbose=True,
            )
            
            scheduler_d = {
                "scheduler": self.scheduler,
                "monitor": "val_loss"
            }
            
        else:
            raise NotImplementedError(f'NotImplemented scheduler: {self.scheduler_name}')
            
    
        return scheduler_d
    
    
    @torch.jit.ignore
    def get_trainable_weights(self, verbose=True):
        trainable_params_v = [p for p in self.parameters() if p.requires_grad ]
        
        if verbose:
            n_w = 0
            for p in trainable_params_v:
                n_w += np.prod(p.shape)

            print(f' - ArcfaceMultiGPU: total trainable weights: {n_w/1e6:0.03} M')

            
        return trainable_params_v
    
    
    @torch.jit.ignore
    def build_optimizer(self):
        if self.optimizer_name is None:
            return None
        
        param_v = list( self.get_trainable_weights() )
        
        if self.optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(
                param_v,
                lr=self.init_lr,
                weight_decay=self.weight_decay,
            )
                
                
        elif self.optimizer_name.lower() == 'sgd': 
            self.optimizer = optim.SGD(
                param_v,
                lr=self.init_lr,
                weight_decay=self.weight_decay,
            )
            
        else:
            raise NotImplementedError(f'NotImplemented optimizer: {self.optimizer_name}')
        
    
        opt_d = {
            "optimizer": self.optimizer,
        }
        
        if self.use_scheduler:
            opt_d["lr_scheduler"] = self._build_scheduler()
        
        return opt_d
    
    
    
# m = ArcfaceMultiGPU(
#     args
# )

# embed = torch.rand(50, args.embedding_size)
# labels = torch.ones(50, dtype=torch.long)
# loss, output, cos_theta = m(embed, labels)

# loss


# In[13]:


class Arcface(nn.Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599    
    def __init__(self, args):
        super().__init__()
        
        self.args           = args
        self.classnum       = args.arc_classnum
        self.arc_bottleneck = args.arc_bottleneck
        self.embedding_size = args.embedding_size
        self.s              = args.arc_s  # the margin value, default is 0.5
        self.m              = args.arc_m  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        
        self.use_bottleneck = (self.arc_bottleneck is not None)

        if self.use_bottleneck:
            self.kernel_A = nn.Parameter(
                torch.Tensor(self.embedding_size, self.arc_bottleneck)
            )
            self.kernel_A.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
            
            self.kernel_B = nn.Parameter(
                torch.Tensor(self.arc_bottleneck, self.classnum)
            )
            self.kernel_B.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
            
        else:
            self.kernel = nn.Parameter(
                torch.Tensor(self.embedding_size, self.classnum)
            )
            self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)


        self.cos_m = np.cos(self.m)
        self.sin_m = np.sin(self.m)
        
        self.mm = (self.sin_m * self.m) # issue 1
        self.threshold = np.cos(np.pi - self.m)
        
        
        self.criterion = nn.CrossEntropyLoss(
                weight=None,
                reduction='mean',
            )
        return None
    
    @torch.jit.ignore
    def build_optimizer(self):
        return None
    
    def project(self, embbedings):
        # weights norm
        nB = len(embbedings)
        if self.use_bottleneck:
            kernel_norm = l2_norm(self.kernel_A @ self.kernel_B, axis=0)
        else:
            kernel_norm = l2_norm(self.kernel, axis=0)
        # cos(theta+m)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        output = self.s * cos_theta
        return output
        
        
    def forward(self, embbedings, label):
        # weights norm
        nB = len(embbedings)
        
        if self.use_bottleneck:
            kernel_norm = l2_norm(self.kernel_A @ self.kernel_B, axis=0)
        else:
            kernel_norm = l2_norm(self.kernel, axis=0)
            
        # cos(theta+m)

        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        
        cos_theta_2 = torch.pow(cos_theta, 2).type(cos_theta.dtype)
        sin_theta_2 = 1.0 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        
        # this condition controls the theta+m should in range [0, pi]
        #  0<=theta+m<=pi
        # -m<=theta<=pi-m
        
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        
        keep_val = (cos_theta - self.mm) # when theta not in [0,pi], use cosface instead
        
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0 # a little bit hacky way to prevent in_place operation on cos_theta
        
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s # scale up in order to make softmax work, first introduced in normface
        
        loss = self.criterion(
            output,
            label,
        )
        
        return loss, output, cos_theta
    
    @torch.jit.ignore
    def get_trainable_weights(self, verbose=True):
        trainable_params_v = [p for p in self.parameters() if p.requires_grad ]
        
        if verbose:
            n_w = 0
            for p in trainable_params_v:
                n_w += np.prod(p.shape)

            print(f' - Arcface: total trainable weights: {n_w/1e6:0.03} M')

            
        return trainable_params_v
# m = Arcface(args)


# In[14]:


class GeM(nn.Module):
    # https://towardsdatascience.com/review-fpn-feature-pyramid-network-object-detection-262fc7482610
    def __init__(self, p=3, eps=1e-6, optimizable_p=True, flatten=True, start_dim=1, end_dim=-1):
        super().__init__()

        self.flatten = flatten
        
        self.p = torch.ones(1) * p
        if optimizable_p:
            self.p = nn.Parameter(self.p)

        if self.flatten:
            self.outlayer = nn.Flatten(start_dim, end_dim)
            
        else:
            self.outlayer = nn.Identity()
        
        self.eps = eps
        return None

    def forward(self, x):
        y = F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))).pow(1.0/self.p)
        
        y = self.outlayer(y)
        return y

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'
    
    
    
class HighWayLayer(nn.Module):
    def __init__(
        self,
        n_features=128,
    ):
        super().__init__() 
        
        self.T = nn.Sequential(
                nn.Linear(
                    in_features=n_features,
                    out_features=n_features
                ),
                nn.Sigmoid())
            
        self.H = nn.Linear(
            in_features=n_features,
            out_features=n_features
        )
        
        return None
    
    
    def forward(self, x):
        y = x * (1.0 - self.T(x)) + self.H(x) * self.T(x)
        return y


# In[15]:


class L2NormLayer(nn.Module):
    def __init__(self, axis=1, eps=1e-6):
        super().__init__()
        self.axis = axis
        self.eps = eps
        return None
    
    def forward(self, x):
        norm = torch.norm(x, 2, self.axis, True) + self.eps
        output = torch.div(x, norm)
        return output

def flip(tensor, dim=1):
    """ Just flip a tensor dim."""
    fliped_idx    = torch.arange(tensor.size(dim)-1, -1, -1).long().to(tensor.device)
    fliped_tensor = tensor.index_select(dim, fliped_idx)
    return fliped_tensor


def cos_decay(start_val=1.0, end_val=1e-4, steps=100):
    return lambda x: ((1 - np.cos(x * np.pi / steps)) / 2) * (end_val - start_val) + start_val

def linear_warmup(start_val=1e-4, end_val=1.0, steps=5):
    return lambda x: x / steps * (end_val - start_val) + start_val  # linear

def scheduler_lambda(lr_frac=1e-4, warmup_epochs=5, cos_decay_epochs=60):
    if warmup_epochs > 0:
        lin = linear_warmup(start_val=lr_frac, end_val=1.0, steps=warmup_epochs)
        
    cos = cos_decay(start_val=1.0, end_val=lr_frac, steps=cos_decay_epochs)
    
    def f(x):
        if x < warmup_epochs:
            return lin(x)
        
        elif x <= (warmup_epochs + cos_decay_epochs):
            return cos(x - warmup_epochs)
        
        else:
            return lr_frac
        
    return f


class FacebookModel(pl.LightningModule):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        
        self.args = args
        
        # # # # # # # # # # # Arguments # # # # # # # # # # # 
        self.model_resolution     = args.model_resolution
        self.embedding_size       = args.embedding_size
        self.backbone_name        = args.backbone_name
        self.pretrained_bb        = args.pretrained_bb
        self.backbone_gpus        = self.args.BACKBONE_GPUS
        self.neck_type            = args.neck_type
        self.eps                  = args.eps
        self.init_lr              = args.init_lr
        
        self.use_scheduler        = args.use_scheduler
        self.scheduler_name       = args.scheduler_name
        self.sched_factor         = args.sched_factor
        self.sched_patience       = args.sched_patience
        self.n_warmup_epochs      = args.n_warmup_epochs
        self.n_decay_epochs       = args.n_decay_epochs
        self.lr_prop              = args.lr_prop
        
        self.optimizer_name       = args.optimizer_name
        self.clip_grad_norm       = args.clip_grad_norm
        self.weight_decay         = args.weight_decay
        self.n_steps_grad_update  = args.n_steps_grad_update
        self.start_ckpt           = args.start_ckpt
        self.checkpoint_base_path = args.checkpoint_base_path
        
        self.criterion_name       = args.criterion_name
        self.arc_m                = args.arc_m
        self.arc_s                = args.arc_s
        self.arc_classnum         = args.arc_classnum
        self.arc_bottleneck       = args.arc_bottleneck
        self.arc_weight_decay     = args.arc_weight_decay
        
        
        self.do_ref_trn           = args.do_ref_trn
        self.ref_trn_n_epochs     = args.ref_trn_n_epochs
        
        self.use_GeM              = args.use_GeM
        self.GeM_p                = args.GeM_p
        self.GeM_opt_p            = args.GeM_opt_p
        self.model_name           = args.model_name
        # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        
        
        self.optimizer = None
        self.scheduler = None
        
        self._create_backbone()
        self._set_criterion()
        
        
        self.trn_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        
        self.trn_acc(torch.zeros(1), torch.ones(1, dtype=torch.int64))
        self.val_acc(torch.zeros(1), torch.ones(1, dtype=torch.int64))
        
        self.trn_acc.reset()
        self.val_acc.reset()
        

        self.trn_ref_acc = torchmetrics.Accuracy()
        self.trn_qry_acc = torchmetrics.Accuracy()

        self.val_ref_acc = torchmetrics.Accuracy()
        self.val_qry_acc = torchmetrics.Accuracy()


        self.trn_ref_acc(torch.zeros(1), torch.ones(1, dtype=torch.int64))
        self.trn_qry_acc(torch.zeros(1), torch.ones(1, dtype=torch.int64))

        self.val_ref_acc(torch.zeros(1), torch.ones(1, dtype=torch.int64))
        self.val_qry_acc(torch.zeros(1), torch.ones(1, dtype=torch.int64))

        self.trn_ref_acc.reset()
        self.trn_qry_acc.reset()

        self.val_ref_acc.reset()
        self.val_qry_acc.reset()
            
        
        if self.start_ckpt is not None:
            self.restore_checkpoint(self.start_ckpt)
        
        # Model Summary
        self.calc_total_weights()
        
        return None
    
    
    @torch.jit.ignore
    def get_trainable_weights(self, verbose=True):
        trainable_params_v = [p for p in self.backbone.parameters() if p.requires_grad ]
        
        if verbose:
            n_w = 0
            for p in trainable_params_v:
                n_w += np.prod(p.shape)

            print(f' - Backbone: total trainable weights: {n_w/1e6:0.03} M')

            
        return trainable_params_v
    
    
    @torch.jit.ignore
    def _build_scheduler(self, LAST_EPOCH=None):
        if self.scheduler_name.lower() == 'LinearWarmupCosineAnnealingLR'.lower():
            self.scheduler = pl_bolts.optimizers.LinearWarmupCosineAnnealingLR(
                optimizer=self.optimizer,                                                               
                warmup_epochs=self.n_warmup_epochs,
                max_epochs=self.n_decay_epochs+self.n_warmup_epochs,
                warmup_start_lr=10 * self.lr_prop * self.init_lr,
                eta_min=self.lr_prop * self.init_lr,
                last_epoch=-1,
            )
            
            scheduler_d = {
                "scheduler": self.scheduler,
            }
            
        elif self.scheduler_name.lower() == 'ReduceLROnPlateau'.lower():
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.sched_factor,
                patience=self.sched_patience,
                verbose=True,
            )
            
            scheduler_d = {
                "scheduler": self.scheduler,
                "monitor": "val_loss"
            }
            
        else:
            raise NotImplementedError(f'NotImplemented scheduler: {self.scheduler_name}')
            
    
        return scheduler_d
    
    
    @torch.jit.ignore
    def build_optimizer(self):
        params_v = self.get_trainable_weights(verbose=True)
        
        
        param_id_v = [id(p) for p in params_v]
        
        if self.optimizer_name.lower() == 'adam':
            if self.weight_decay > 0.0:
                pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
                for k, v in self.named_modules():
                    if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                        if id(v.bias) in param_id_v:
                            pg2.append(v.bias)  # biases, no decay
                        else:
                            v.bias.requires_grad_(False)
                            
                    if isinstance(v, nn.BatchNorm2d):
                        if id(v.weight) in param_id_v:
                            pg0.append(v.weight)  # weights, no decay
                        else:
                            v.weight.requires_grad_(False)
                        
                    elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                        if id(v.weight) in param_id_v:
                            pg1.append(v.weight)  # weights, decay
                        else:
                            v.weight.requires_grad_(False)
                
                if len(pg0) > 0:
                    # Weights Nodecay
                    self.optimizer = optim.Adam(pg0, lr=self.init_lr)
                else:
                    self.optimizer = optim.Adam(pg2, lr=self.init_lr)
                    pg2 = []
                
                if len(pg1) > 0:
                    # Weights Decay
                    self.optimizer.add_param_group({'params': pg1, 'weight_decay': self.weight_decay})
                
                if len(pg2) > 0:
                    # Biasese NoDecay
                    self.optimizer.add_param_group({'params': pg2})

                del pg0, pg1, pg2

            else:   
                self.optimizer = optim.Adam(
                    params_v,
                    lr=self.init_lr,
                    weight_decay=self.weight_decay,
                )
                
                
        elif self.optimizer_name.lower() == 'sgd':
            if self.weight_decay > 0.0:
                pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
                for k, v in self.named_modules():
                    if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                        if id(v.bias) in param_id_v:
                            pg2.append(v.bias)  # biases, no decay
                        else:
                            v.bias.requires_grad_(False)
                            
                    if isinstance(v, nn.BatchNorm2d):
                        if id(v.weight) in param_id_v:
                            pg0.append(v.weight)  # weights, no decay
                        else:
                            v.weight.requires_grad_(False)
                        
                    elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                        if id(v.weight) in param_id_v:
                            pg1.append(v.weight)  # weights, decay
                        else:
                            v.weight.requires_grad_(False)
                
                if len(pg0) > 0:
                    # Weights Nodecay
                    self.optimizer = optim.SGD(pg0, lr=self.init_lr)
                else:
                    self.optimizer = optim.SGD(pg2, lr=self.init_lr)
                    pg2 = []
                
                if len(pg1) > 0:
                    # Weights Decay
                    self.optimizer.add_param_group({'params': pg1, 'weight_decay': self.weight_decay})
                
                if len(pg2) > 0:
                    # Biasese NoDecay
                    self.optimizer.add_param_group({'params': pg2})

                del pg0, pg1, pg2

            else:   
                self.optimizer = optim.SGD(
                    params_v,
                    lr=self.init_lr,
                    weight_decay=self.weight_decay,
                )
            
        else:
            raise NotImplementedError(f'NotImplemented optimizer: {self.optimizer_name}')
        
        
        criterion_param_v = self.criterion.get_trainable_weights(verbose=True)
        for p in criterion_param_v:
            self.optimizer.add_param_group(
                {'params': p, 'weight_decay': self.arc_weight_decay}
            )
        
        opt_d = {
            "optimizer": self.optimizer,
        }
        
        if self.use_scheduler:
            opt_d["lr_scheduler"] = self._build_scheduler()
        
        
        return opt_d
    
    
    @torch.jit.ignore
    def _set_criterion(self):
        
        if self.criterion_name.lower() == 'arcface':
            self.criterion = Arcface(
                self.args
            )
            
        elif self.criterion_name.lower() == 'arcface_multigpu':
            self.criterion = ArcfaceMultiGPU(self.args)
            
        else:
            raise NotImplementedError('Not Implemented Criterion: {self.criterion}')
            
        return None
    
    
    @torch.jit.ignore
    def find_last_saved_ckpt(self):
        def get_version_from_path(fp):
            i_v = fp.find('version_')
            vn = ''
            for i in range(i_v+8, len(fp)):
                if fp[i].isdigit():
                    vn += fp[i]
                else:
                    break

            if len(vn) > 0:
                vn = int(vn)

            else:
                vn = -1

            return vn
        
        all_last_ckpt_v = glob.glob(
            os.path.join( model.checkpoint_base_path, '**/last.ckpt'),
            recursive=True
            )

        if len(all_last_ckpt_v) > 0:
            all_ckpt_version_v = [get_version_from_path(fp) for fp in all_last_ckpt_v]
            idx = np.argmax( all_ckpt_version_v )

            return all_last_ckpt_v[idx]

        else:
            return None
    
    
    @torch.jit.ignore
    def restore_checkpoint(
        self, 
        PATH=None, 
        load_optimizer=False,
        load_scheduler=False,
        verbose=True):
        
        
        if PATH is None:
            PATH = self.find_last_saved_ckpt()
            
        checkpoint = torch.load(
            PATH,
            map_location=self.device,)

        if 'epoch' in checkpoint.keys():
            epoch = checkpoint['epoch']
            
        if 'global_step' in checkpoint.keys():
            global_step = checkpoint['global_step']
            
        saved_state_dict = checkpoint['state_dict']
        
        current_state_dict = self.state_dict()
        new_state_dict = OrderedDict()
        for key in current_state_dict.keys():
            
            key_no_dp_m0 = 'ref_' + key.replace('module.0.', '')
            key_no_dp_m1 = key.replace('backbone.module.1', 'ref_neck.0')
            key_no_dp_m2 = key.replace('backbone.0', 'ref_backbone')
            key_no_dp_m3 = key.replace('backbone.1', 'ref_neck.0')
        
            if (key in saved_state_dict.keys()) and (saved_state_dict[key].shape == current_state_dict[key].shape):
                new_state_dict[key] = saved_state_dict[key]
            
            elif (key_no_dp_m0 in saved_state_dict.keys()) and (saved_state_dict[key_no_dp_m0].shape == current_state_dict[key].shape):
                new_state_dict[key] = saved_state_dict[key_no_dp_m0]
            
            elif (key_no_dp_m1 in saved_state_dict.keys()) and (saved_state_dict[key_no_dp_m1].shape == current_state_dict[key].shape):
                new_state_dict[key] = saved_state_dict[key_no_dp_m1]

            elif (key_no_dp_m2 in saved_state_dict.keys()) and (saved_state_dict[key_no_dp_m2].shape == current_state_dict[key].shape):
                new_state_dict[key] = saved_state_dict[key_no_dp_m2]
            
            elif (key_no_dp_m3 in saved_state_dict.keys()) and (saved_state_dict[key_no_dp_m3].shape == current_state_dict[key].shape):
                new_state_dict[key] = saved_state_dict[key_no_dp_m3]
                
            else:
                load_optimizer = False
                if key not in saved_state_dict.keys():
                    print(f' - WARNING: key="{key}" not found in saved checkpoint. Weights will not be loaded.', file=sys.stderr)
                else:
                    s0 = tuple(saved_state_dict[key].shape)
                    s1 = tuple(current_state_dict[key].shape)
                    print(f' - WARNING: shapes mismatch in "{key}": {s0} vs {s1}. Weights will not be loaded.', file=sys.stderr)
                
                new_state_dict[key] = current_state_dict[key]
        
        self.load_state_dict( new_state_dict )
        
        if self.optimizer is not None:
            if load_optimizer:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_states'])
                except Exception as e:
                    print(' - WARNING: ERROR while loading the optimizer. The optimizer will be reseted.', file=sys.stderr)
                    self.build_optimizer()
        
                
        if self.scheduler is not None:
            if load_scheduler:
                try:
                    self.scheduler.load_state_dict(checkpoint['lr_schedulers'])
                    
                except Exception as e:
                    print(' - WARNING: ERROR while loading the scheduler. The Scheduler will be reseted.', file=sys.stderr)
                    
                    self._build_scheduler(LAST_EPOCH=epoch)
                    
        if verbose:
            print(f' - Restored checkpoint: {PATH}.')
        
        return checkpoint 
    
    
    @torch.jit.ignore
    def calc_total_weights(self, verbose=True):
        n_w = 0
        for p in self.parameters():
            n_w += np.prod(p.shape, dtype=np.int64)
        
        if verbose:
            print(' - Total weights: {:0.02f}M'.format(n_w/1e6))
        
        return n_w
    
    
    @torch.jit.export
    def forward(
        self,
        img,
    ):
        y = self.backbone(img)
        
        return y
    
   
    @torch.jit.export
    def arcface_step(self, data, batch_idx):
        
        ref_img = data['ref_img']
        qry_img = data['qry_img']
        cls     = data['cls']
        
        e_ref = self.forward(ref_img)
        e_qry = self.forward(qry_img)
        
        if self.criterion is not None:
            loss_ref, output_ref, cos_theta_ref = self.criterion(e_ref, cls)
            loss_qry, output_qry, cos_theta_qry = self.criterion(e_qry, cls)
    
        else:
            loss_ref, output_ref, cos_theta_ref = criterion(e_ref, cls)
            loss_qry, output_qry, cos_theta_qry = criterion(e_qry, cls)
            
        
        corrects_ref = cos_theta_ref.argmax(dim=-1).type(torch.float32)
        corrects_qry = cos_theta_qry.argmax(dim=-1).type(torch.float32)
        
        loss = 0.5 * (loss_ref + loss_qry)
        
        corrects     = (corrects_ref == corrects_qry).type(torch.float32).detach().cpu()
        corrects_qry = (cls == corrects_qry).type(torch.float32).detach().cpu()
        corrects_ref = (cls == corrects_ref).type(torch.float32).detach().cpu()
        
        return loss, corrects, corrects_qry, corrects_ref
    
    
    @torch.jit.export
    def arcface_qry_step(self, data, batch_idx):
        """ returns: (loss, corrects_qry) """
        
        qry_img = data['qry_img']
        cls = data['cls']
        
        e_qry = self.forward(qry_img)
        
        loss_qry, output_qry, cos_theta_qry = self.criterion(e_qry, cls)
        
        corrects_qry = cos_theta_qry.argmax(dim=-1).type(torch.float32)
        
        loss = loss_qry
        
        corrects_qry = (cls == corrects_qry).type(torch.float32).detach().cpu()
        
        return loss, corrects_qry
    
    
    @torch.jit.export
    def training_step(self, data, batch_idx):
        """ Evals arcface_qry_step and returns loss """
        
        if self.do_ref_trn and ( (self.trainer.current_epoch % self.ref_trn_n_epochs) == 0 ):
            loss, corrects, corrects_qry, corrects_ref = self.arcface_step(data, batch_idx)


            self.log('trn_acc',     self.trn_acc(corrects, torch.ones_like(corrects, dtype=torch.int64)),     on_step=True, on_epoch=True, rank_zero_only=True, sync_dist=True)
            self.log('trn_qry_acc', self.trn_qry_acc(corrects_qry, torch.ones_like(corrects_qry, dtype=torch.int64)), on_step=True, on_epoch=True, rank_zero_only=True, sync_dist=True)
            self.log('trn_ref_acc', self.trn_ref_acc(corrects_ref, torch.ones_like(corrects_ref, dtype=torch.int64)), on_step=True, on_epoch=True, rank_zero_only=True, sync_dist=True)


        else:
            loss, corrects_qry = self.arcface_qry_step(data, batch_idx)

            self.log('trn_qry_acc', self.trn_qry_acc(corrects_qry, torch.ones_like(corrects_qry, dtype=torch.int64)), on_step=True, on_epoch=True, rank_zero_only=True, sync_dist=True)
        
        self.log('trn_loss', loss.detach().cpu(), on_step=True, on_epoch=True, rank_zero_only=True, sync_dist=True)
        
        if self.trainer.is_global_zero:
            self.log('lr',  self.get_lr(), on_step=True, on_epoch=False, rank_zero_only=True, sync_dist=True)
        
        if torch.isnan(loss) or torch.isnan(loss):
            raise Exception(f"Loss is NaN or Inf. loss={loss}")
            
        return loss
    

    @torch.jit.export
    def validation_step(self, data, batch_idx):
        
        if True:
            loss, corrects, corrects_qry, corrects_ref = self.arcface_step(data, batch_idx)

            self.log(
                'val_acc',     
                self.val_acc(corrects, torch.ones_like(corrects, dtype=torch.int64)),     
                on_step=True, on_epoch=True, rank_zero_only=True, sync_dist=True)
            self.log(
                'val_qry_acc', 
                self.val_qry_acc(corrects_qry, torch.ones_like(corrects_qry, dtype=torch.int64)), 
                on_step=True, on_epoch=True, rank_zero_only=True, sync_dist=True)
            self.log(
                'val_ref_acc', 
                self.val_ref_acc(corrects_ref, torch.ones_like(corrects_ref, dtype=torch.int64)), 
                on_step=True, on_epoch=True, rank_zero_only=True, sync_dist=True)

        else:
            loss, corrects_qry = self.arcface_qry_step(data, batch_idx)

            self.log(
                'val_qry_acc', 
                self.val_qry_acc(corrects_qry, torch.ones_like(corrects_qry, dtype=torch.int64)), 
                on_step=True, on_epoch=True, rank_zero_only=True, sync_dist=True)
                
                

        if self.trainer.is_global_zero:
            self.log('val_loss', loss.detach().cpu(), on_step=True, on_epoch=True, rank_zero_only=True, sync_dist=True)
            
            
        return None
    
    @torch.jit.ignore
    def configure_optimizers(self):
        print(' - Setting up the optimizer.')
        return self.build_optimizer()
        
    
    @torch.jit.ignore
    def set_lr(self, new_lr=1e-3):
        for param_group in self.optimizer.param_groups:
            if 'lr' in param_group.keys():
                param_group['lr'] = new_lr
        
        self.lr = new_lr
        
        return None
    
    
    @torch.jit.ignore
    def get_lr(self):
        if self.optimizer is not None:
            to_ret = None
            
            # Checking lr of optimizer.
            for param_group in self.optimizer.param_groups:
                if to_ret is None:
                    to_ret = param_group['lr']
            
        else:
            to_ret = self.lr
        
        return to_ret
    
    
    @torch.jit.ignore
    def _create_backbone(self):
        
        def remove_head(backbone):
            if 'fc' in dir(backbone):
                n_features = backbone.fc.in_features
                del(backbone.fc)
                backbone.fc = nn.Identity()

            elif 'classifier' in dir(backbone):
                n_features = backbone.classifier.in_features
                del(backbone.classifier)
                backbone.classifier = nn.Identity()

            else:
                raise NotImplementedError(f'Does the BackBone have a head?')

            return n_features
        
        
        backbone = timm.create_model(
            self.backbone_name,
            pretrained=self.pretrained_bb
        )
        
        n_features = remove_head(backbone)

        
        if self.use_GeM:
            del(backbone.global_pool)
            backbone.global_pool = GeM(
                p=self.GeM_p,
                eps=self.eps,
                optimizable_p=self.GeM_opt_p,
                flatten=True,
                start_dim=1,
                end_dim=-1)
         
        
        if self.neck_type == 'A':
            self.backbone = nn.Sequential(
                backbone,
                nn.Linear(
                    n_features,
                    self.embedding_size,
                    True
                ),

                L2NormLayer(
                    axis=1,
                    eps=self.eps,
                ),
            )

            
        elif self.neck_type == 'B':
            self.backbone = nn.Sequential(
                backbone,
                nn.Linear(
                    n_features,
                    self.embedding_size,
                    True
                ),
                nn.PReLU(),
                L2NormLayer(
                    axis=1,
                    eps=self.eps,
                ),
            )
                
        elif self.neck_type == 'C':
            self.backbone = nn.Sequential(
                backbone,
                HighWayLayer(
                    n_features
                ),
                nn.Linear(
                    n_features,
                    self.embedding_size,
                    True
                ),
                nn.PReLU(),
                L2NormLayer(
                    axis=1,
                    eps=self.eps,
                ),
            )
            
        elif self.neck_type == 'D':
            self.backbone = nn.Sequential(
                backbone,
                HighWayLayer(
                    n_features
                ),
                nn.PReLU(),
                nn.Linear(
                    n_features,
                    self.embedding_size,
                    True
                ),
                L2NormLayer(
                    axis=1,
                    eps=self.eps,
                ),
            )

        else:
            raise NotImplementedError(f'neck_type = "{self.neck_type}"')
            
        if len(self.backbone_gpus) > 1 and (self.criterion_name.lower() == 'arcface_multigpu'):
            print(' - Using DataParallel, devices: ', [torch.device(d) for d in self.backbone_gpus])
            self.backbone = nn.parallel.DataParallel(
                self.backbone,
                device_ids=self.backbone_gpus,
            )
            
        return None
    
    
    
    @torch.jit.ignore
    def predict_embedding(
        self,
        img,
        do_simple_augmentation=False,
        calc_triplet_cls=True
    ):
        
        embed = self.forward(img)
        
        if do_simple_augmentation:
            img = flip(img, dim=2) 
            embed_aug = self.forward(img)
            embed  = l2_norm( embed + embed_aug )  
            
        
        if calc_triplet_cls:
            projection = self.criterion.project(embed)
            return embed, projection
        
        else:
            return embed
    
    
    def to(self, device):
        if self.criterion_name.lower() == 'arcface_multigpu':
            print(" - Setting Backbone's device:", device)
            ret = self.backbone.to(device)
            self.criterion._set_devices()
        
        else:
            print(" - Setting model's device:", device)
            ret = super().to(device)
            
        return ret
    
    
    def load_state_dict(self, state_dict):
        print(' - Loading model state_dict.')
#         for k in state_dict.keys():
#             print(k, state_dict[k].device)
        return super().load_state_dict(state_dict)


# In[16]:


# model = FacebookModel(args)

