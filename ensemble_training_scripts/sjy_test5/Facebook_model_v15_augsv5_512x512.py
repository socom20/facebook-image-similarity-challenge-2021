#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys, gc
import time
import glob
import pickle
import copy
import json
import random
from collections import OrderedDict, namedtuple
import multiprocessing

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

from IPython.display import display, clear_output

import faiss

from AugsDS_v5 import *


# In[2]:


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# In[3]:


print('Detected devices:' )
for i in range( torch.cuda.device_count() ):
    print( f' *', str( torch.cuda.get_device_properties(i))[1:] )


# # Global Configuration

# In[4]:


BATCH_SIZE   = 8
N_WORKERS    = 12
N_GPUS       = 1             # Sets the number of GPUs
DS_INPUT_DIR = f'./all_datasets/dataset'  # Path where "query_images", "reference_images" and "training_images" are located
CPY_QRY2REF  = False

# DATASET_WH = (384, 384)
DATASET_WH = (512, 512)

#OUTPUT_WH = (224, 224)
# OUTPUT_WH = (384, 384)
OUTPUT_WH = (512, 512)


# In[5]:


while DS_INPUT_DIR[-1] in ['/', r'\\']:
    DS_INPUT_DIR = DS_INPUT_DIR[:-1]

# Path where the rescaled images will be saved
DS_DIR = f'{DS_INPUT_DIR}_jpg_{DATASET_WH[0]}x{DATASET_WH[1]}'



# In[7]:


class Args():
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
    
    


    
    
class ArgsT15_EffNetV2L(Args):
    OUTPUT_WH            = OUTPUT_WH
    DATASET_WH           = DATASET_WH
    BATCH_SIZE           = BATCH_SIZE
    N_WORKERS            = N_WORKERS
    N_GPUS               = N_GPUS
    DS_INPUT_DIR         = DS_INPUT_DIR
    DS_DIR               = DS_DIR
    
    img_norm_type        = ['simple', 'imagenet', 'centered'][1]
    model_resolution     = OUTPUT_WH[::-1]
    embedding_size       = 128 * 4
    backbone_name        = ['efficientnetv2_rw_s', 'efficientnetv2_rw_m', 'tf_efficientnetv2_l_in21ft1k'][2]
    pretrained_bb        = True
    duplicate_backbone   = False
    duplicate_neck       = False
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
    criterion_name       = ['arcface', 'triplet', 'tripletarc'][0]
    
    
    do_ref_trn           = True
    ref_trn_n_epochs     = 5
    
    arc_classnum         = 1_000_000 - 20_000
    arc_m                = 0.4
    arc_s                = 40.0
    arc_gamma            = 3.0
    use_GeM              = True
    GeM_p                = 3.0
    GeM_opt_p            = True
    checkpoint_base_path = './TEST15_arcface'
    model_name           = 'arc_model'
    
    precision            = [32, 16][1]
    seed                 = 1578
    
    accelerator          = ['dp', 'ddp'][1]
    
    

args = ArgsT15_EffNetV2L()



print(args)


# # Setting SEED

# In[8]:


def seed_everything(seed=1427):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    return None


# In[9]:


seed_everything(seed=args.seed)


# # Data Source

# In[10]:


if not os.path.exists(args.DS_DIR):
    assert os.path.exists(args.DS_INPUT_DIR), f'DS_INPUT_DIR not found: {args.DS_INPUT_DIR}'
    
    resize_dataset(
        ds_input_dir=args.DS_INPUT_DIR,
        ds_output_dir=args.DS_DIR,
        output_wh=args.OUTPUT_WH,
        output_ext='jpg',
        num_workers=args.N_WORKERS,
        verbose=False,
    )
    


# In[11]:


print('Paths:')
print(' - DS_INPUT_DIR:', args.DS_INPUT_DIR)
print(' - DS_DIR:      ', args.DS_DIR)

assert os.path.exists(args.DS_DIR), f'DS_DIR not found: {args.DS_DIR}'


# In[12]:


try:
    public_gt = pd.read_csv( os.path.join(args.DS_DIR, 'public_ground_truth.csv') )
    
except:
    public_gt = pd.read_csv( os.path.join(args.DS_INPUT_DIR, 'public_ground_truth.csv') )


# # Synthetic DataSets

# In[13]:


if args.criterion_name == 'arcface':
    n_trn_samples = args.arc_classnum

    trn_ref_samples_id_v = np.array([f'T{i:06d}' for i in range(0, n_trn_samples)])
    trn_neg_samples_id_v = None

    val_ref_samples_id_v = np.array([f'T{i:06d}' for i in range(0, n_trn_samples)])
    val_neg_samples_id_v = None
    
    trn_overlay_img_ids_v = np.array([f'T{i:06d}' for i in range(n_trn_samples, 1_000_000)])
    val_overlay_img_ids_v = np.array([f'T{i:06d}' for i in range(n_trn_samples, 1_000_000)])
    
else:
    n_trn_samples = 1_000_000 - 25_000
    n_val_samples = 25_000

    trn_ref_samples_id_v = np.array([f'T{i:06d}' for i in range(0, n_trn_samples)])
    trn_neg_samples_id_v = None

    val_ref_samples_id_v = np.array([f'R{i:06d}' for i in range(0, n_val_samples)])
    val_neg_samples_id_v = None
    
    trn_overlay_img_ids_v = np.array([f'T{i:06d}' for i in range(n_trn_samples, 1_000_000)])
    val_overlay_img_ids_v = np.array([f'R{i:06d}' for i in range(n_val_samples, 1_000_000)])


ds_trn = FacebookSyntheticDataset(
    ref_samples_id_v=trn_ref_samples_id_v,
    neg_samples_id_v=trn_neg_samples_id_v,

    do_ref_augmentation=True,
    ds_dir=args.DS_DIR,
    output_wh=args.OUTPUT_WH,
    max_retries=10,
    channel_first=True,
    return_neg_img=(args.criterion_name.lower() != 'arcface'),
    overlay_img_ids_v=trn_overlay_img_ids_v,
    norm_type=args.img_norm_type,
    verbose=True,
)

# ds_trn.plot_sample(8)


ds_val = FacebookSyntheticDataset(
    ref_samples_id_v=val_ref_samples_id_v,
    neg_samples_id_v=val_neg_samples_id_v,

    do_ref_augmentation=True,
    ds_dir=args.DS_DIR,
    output_wh=args.OUTPUT_WH,
    max_retries=10,
    channel_first=True,
    return_neg_img=(args.criterion_name.lower() != 'arcface'),
    overlay_img_ids_v=val_overlay_img_ids_v,
    norm_type=args.img_norm_type,
    verbose=True,
)

# ds_val.plot_sample(8)


# # Inference Datasets

# In[14]:


ds_qry_full = FacebookDataset(
    samples_id_v=[f'Q{i:05d}' for i in range(50_000)],
    do_augmentation=False,
    ds_dir=args.DS_DIR,
    output_wh=args.OUTPUT_WH,
    channel_first=True,
    norm_type= args.img_norm_type,
    verbose=True,
)
# ds_qry_full.plot_sample(4)


ds_ref_full = FacebookDataset(
    samples_id_v=[f'R{i:06d}' for i in range(1_000_000)],
    do_augmentation=False,
    ds_dir=args.DS_DIR,
    output_wh=args.OUTPUT_WH,
    channel_first=True,
    norm_type=args.img_norm_type,
    verbose=True,
)
# ds_ref_full.plot_sample(4)


ds_trn_full = FacebookDataset(
    samples_id_v=[f'T{i:06d}' for i in range(1_000_000)],
    do_augmentation=False,
    ds_dir=args.DS_DIR,
    output_wh=args.OUTPUT_WH,
    channel_first=True,
    norm_type=args.img_norm_type,
    verbose=True,
)
# ds_trn_full.plot_sample(4)


# # Model

# In[15]:


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
    
class Arcface(nn.Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599    
    def __init__(self, embedding_size=512, classnum=10, s=64., m=0.5, gamma=1.0):
        super(Arcface, self).__init__()
        self.classnum = classnum
        self.embedding_size = embedding_size
        self.gamma = gamma
        
        
#         self.kernel = torch.tensor( [[np.cos(2 * np.pi / classnum * i), np.sin(2 * np.pi / classnum * i)] for i in range(classnum)], dtype=torch.float32 ).T
#         self.kernel = nn.Parameter(self.kernel, requires_grad=False)
        
        # initial kernel
        self.kernel = nn.Parameter(
            torch.Tensor(embedding_size, classnum)
        )
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
#         nn.init.xavier_normal_(self.kernel)

        self.m = m # the margin value, default is 0.5
        self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        
        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        
        self.mm = (self.sin_m * m) # issue 1
        self.threshold = np.cos(np.pi - m)
        
#         self.criterion = FocalLoss(
#             gamma=self.gamma,
#             positve_weighting=-1
#         )
        
        self.criterion = nn.CrossEntropyLoss(
                weight=None,
                reduction='mean',
            )
        return None
    
    
    def project(self, embbedings):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel, axis=0)
        # cos(theta+m)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        output = self.s * cos_theta
        return output
        
        
    def forward(self, embbedings, label):
        # weights norm
        nB = len(embbedings)
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


# In[16]:


class TripletArcLoss(nn.Module):
    def __init__(self, s=10.0, m=0.5, gamma=2.0):
        super().__init__()
        
        self.m = m # the margin value, default is 0.5
        self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.gamma = gamma  # Focal loss
        
                 
        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        
        self.mm = (self.sin_m * m) # issue 1
        self.threshold = np.cos(np.pi - m)
        
        
        return None
    
    def calc_prob(
        self, 
        y_ref:torch.Tensor, 
        y_qry:torch.Tensor,
    ):
        
        cos_theta = (y_ref * y_qry).sum(axis=-1)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        
        logit = self.s * cos_theta
        return logit.sigmoid()
    
    
    def forward(
        self,
        y_ref:torch.Tensor,
        y_pos:torch.Tensor,
        y_neg:torch.Tensor,
    ):
        # --------------------------------------------------------------------------------
        # Positive Sample
        # --------------------------------------------------------------------------------
        # cos(theta+m)
        cos_theta = torch.sum(y_ref * y_pos, dim=-1)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        
        cos_theta_2 = torch.pow(cos_theta, 2).type(cos_theta.dtype)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        
        # this condition controls the theta+m should in range [0, pi]
        #  0<=theta+m<=pi
        # -m<=theta<=pi-m        
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        
        keep_val = (cos_theta - self.mm) # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        
        logit_pos = self.s * cos_theta_m
        # --------------------------------------------------------------------------------
        
        
        # --------------------------------------------------------------------------------
        # Negative Sample
        # --------------------------------------------------------------------------------
        # cos(theta+m)
        cos_theta = torch.sum(y_ref * y_neg, dim=-1)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        
        logit_neg = self.s * cos_theta
        # --------------------------------------------------------------------------------
        
        
#         loss = \
#         nn.functional.binary_cross_entropy_with_logits(logit_pos, torch.ones_like(logit_pos)) + \
#         nn.functional.binary_cross_entropy_with_logits(logit_neg, torch.zeros_like(logit_neg))
        
        p_pos = logit_pos.sigmoid()
        p_neg = logit_neg.sigmoid()
        
        # Using Focal Loss
        cost  = torch.pow(1.0 - p_pos, self.gamma) * torch.log( p_pos ) + torch.pow(p_neg, self.gamma) * torch.log( 1.0 - p_neg )
        
        loss = - torch.mean(cost, dim=0)
        return loss


# In[17]:


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


# In[18]:


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
        self.duplicate_backbone   = args.duplicate_backbone
        self.duplicate_neck       = args.duplicate_neck
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
        
        self.do_ref_trn           = args.do_ref_trn
        self.ref_trn_n_epochs     = args.ref_trn_n_epochs
        
        self.arc_gamma            = args.arc_gamma
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
        
        if not self.triplet_fw:
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
            
        
        # Moving model to device
        self.to(self.device)
        
        if self.start_ckpt is not None:
            self.restore_checkpoint(self.start_ckpt)
        
        # Model Summary
        self.calc_total_weights()
        
        # Building Optimizers
        self.build_optimizer()
        
        return None
    
    
    @torch.jit.ignore
    def get_trainable_weights(self, verbose=True):
        
        trainable_params_v = [p for p in self.parameters() if p.requires_grad ]
        
        if verbose:
            n_w = 0
            for p in trainable_params_v:
                n_w += np.prod(p.shape)

            print(f' - Total trainable weights: {n_w/1e6:0.03} M')

            
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
    def build_optimizer(self, params_v=None):
        if params_v is None:
            params_v = self.get_trainable_weights(verbose=False)
#             params_v = self.parameters()
        
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
        
    
        opt_d = {
            "optimizer": self.optimizer,
        }
        
        if self.use_scheduler:
            opt_d["lr_scheduler"] = self._build_scheduler()
        
        return opt_d
    
    
    @torch.jit.ignore
    def _set_criterion(self):
        
        if self.criterion_name.lower() == 'triplet':
            self.criterion = torch.nn.TripletMarginLoss(
                margin=self.arc_m,
                p=2.0,
                eps=self.eps,
                swap=False,
                size_average=None,
                reduce=None,
                reduction='mean'
            )
            self.triplet_fw = True
            
        elif self.criterion_name.lower() == 'tripletarc':
            self.criterion = TripletArcLoss(
                s=self.arc_s, 
                m=self.arc_m,
                gamma=self.arc_gamma,
            )
            self.triplet_fw = True
            
        elif self.criterion_name.lower() == 'arcface':
            self.criterion = Arcface(
                s=self.arc_s, 
                m=self.arc_m,
                gamma=self.arc_gamma,
                classnum=args.arc_classnum,
                embedding_size=self.embedding_size,
            )
            self.triplet_fw = False
            
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

        
        epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        saved_state_dict = checkpoint['state_dict']
        
        current_state_dict = self.state_dict()
        new_state_dict = OrderedDict()
        for key in current_state_dict.keys():
            if (key in saved_state_dict.keys()) and (saved_state_dict[key].shape == current_state_dict[key].shape):
                new_state_dict[key] = saved_state_dict[key]

            else:
                load_optimizer = False
                if key not in saved_state_dict.keys():
                    print(f' - WARNING: key="{key}" not found in saved checkpoint.\n   Weights will not be loaded.', file=sys.stderr)
                else:
                    s0 = tuple(saved_state_dict[key].shape)
                    s1 = tuple(current_state_dict[key].shape)
                    print(f' - WARNING: shapes mismatch in "{key}": {s0} vs {s1}.\n   Weights will not be loaded.', file=sys.stderr)
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
    def forward_qry(
        self,
        qry_img,
    ):
        y_qry = self.qry_backbone(qry_img)
        y_qry = self.qry_neck(y_qry)
        return y_qry
    
    @torch.jit.export
    def forward_ref(
        self,
        ref_img,
    ):
        y_ref = self.ref_backbone(ref_img)
        y_ref = self.ref_neck(y_ref)
        
        return y_ref
    
    @torch.jit.export
    def triplet_step(self, data, batch_idx):
        
        ref_img = data['ref_img']
        qry_img = data['qry_img']
        neg_img = data['neg_img']
        
        y_ref = self.forward_ref(ref_img)
        y_qry = self.forward_qry(qry_img)
        y_neg = self.forward_qry(neg_img)
        
        loss = self.criterion(y_ref, y_qry, y_neg)
        
        d2pos = torch.norm(y_ref - y_qry, dim=1)
        d2neg = torch.norm(y_ref - y_neg, dim=1)
        
        corrects = (d2pos < d2neg).type(torch.float32)
        
        return loss, corrects.detach().cpu()
    
    
    @torch.jit.export
    def arcface_step(self, data, batch_idx):
        
        ref_img = data['ref_img']
        qry_img = data['qry_img']
        cls     = data['cls']
        
        e_ref = self.forward_ref(ref_img)
        e_qry = self.forward_qry(qry_img)
        
        loss_ref, output_ref, cos_theta_ref = self.criterion(e_ref, cls)
        loss_qry, output_qry, cos_theta_qry = self.criterion(e_qry, cls)
        
        corrects_ref = cos_theta_ref.argmax(dim=-1).type(torch.float32)
        corrects_qry = cos_theta_qry.argmax(dim=-1).type(torch.float32)
        
        loss = 0.5 * (loss_ref + loss_qry)
        
        corrects     = (corrects_ref == corrects_qry).type(torch.float32).detach().cpu()
        corrects_qry = (cls == corrects_qry).type(torch.float32).detach().cpu()
        corrects_ref = (cls == corrects_ref).type(torch.float32).detach().cpu()
        
        return loss, corrects, corrects_qry, corrects_ref
    
    
    @torch.jit.export
    def arcface_qry_step(self, data, batch_idx):
        
        qry_img = data['qry_img']
        cls = data['cls']
        
        e_qry = self.forward_qry(qry_img)
        
        loss_qry, output_qry, cos_theta_qry = self.criterion(e_qry, cls)
        

        corrects_qry = cos_theta_qry.argmax(dim=-1).type(torch.float32)
        
        loss = loss_qry
        
        corrects_qry = (cls == corrects_qry).type(torch.float32).detach().cpu()
        
        return loss, corrects_qry
    
    
    @torch.jit.export
    def training_step(self, data, batch_idx):
        
        if self.triplet_fw:
            loss, corrects = self.triplet_step(data, batch_idx)
            
#             self.trn_acc(corrects, torch.ones_like(corrects, dtype=torch.int64))
            
            if self.trainer.is_global_zero:
                self.log('trn_acc', self.trn_acc(corrects, torch.ones_like(corrects, dtype=torch.int64)), on_step=True, on_epoch=True, rank_zero_only=True, sync_dist=True)
                
        else:
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
            
        return loss
    

    @torch.jit.export
    def validation_step(self, data, batch_idx):
        
        if self.triplet_fw:
            loss, corrects = self.triplet_step(data, batch_idx)
            
            if self.trainer.is_global_zero:
                self.log(
                    'val_acc',
                    self.val_acc(corrects, torch.ones_like(corrects, dtype=torch.int64)),
                    on_step=True, on_epoch=True, rank_zero_only=True, sync_dist=True)

        else:
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
        
        
        self.ref_backbone = timm.create_model(
            self.backbone_name,
            pretrained=self.pretrained_bb
        )
        
        n_ref_features = remove_head(self.ref_backbone)

        
        if self.use_GeM:
            del(self.ref_backbone.global_pool)
            self.ref_backbone.global_pool = GeM(
                p=self.GeM_p,
                eps=self.eps,
                optimizable_p=self.GeM_opt_p,
                flatten=True,
                start_dim=1,
                end_dim=-1)
        
        
        if not self.duplicate_backbone:
            self.qry_backbone = self.ref_backbone
            n_qry_features = n_ref_features
            
        else:
            self.qry_backbone = timm.create_model(
                self.backbone_name,
                pretrained=self.pretrained_bb
            )
            n_qry_features = remove_head(self.qry_backbone)
            
            if self.use_GeM:
                del(self.qry_backbone.global_pool)
                self.qry_backbone.global_pool = GeM(
                    p=self.GeM_p,
                    eps=self.eps,
                    optimizable_p=self.GeM_opt_p,
                    flatten=True,
                    start_dim=1,
                    end_dim=-1)
        
        
        if self.neck_type == 'A':
            self.ref_neck = nn.Sequential(
                nn.Linear(
                    n_ref_features,
                    self.embedding_size,
                    True
                ),

                L2NormLayer(
                    axis=1,
                    eps=self.eps,
                ),
            )


            if not self.duplicate_neck:
                self.qry_neck = self.ref_neck

            else:
                self.qry_neck = nn.Sequential(
                    nn.Linear(
                        n_qry_features,
                        self.embedding_size,
                        True
                    ),

                    L2NormLayer(
                        axis=1,
                        eps=self.eps,
                    ),
                )
                
        elif self.neck_type == 'B':
            self.ref_neck = nn.Sequential(
                nn.Linear(
                    n_ref_features,
                    self.embedding_size,
                    True
                ),
                nn.PReLU(),
                L2NormLayer(
                    axis=1,
                    eps=self.eps,
                ),
            )

            if not self.duplicate_neck:
                self.qry_neck = self.ref_neck

            else:
                self.qry_neck = nn.Sequential(
                    nn.Linear(
                        n_qry_features,
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
            self.ref_neck = nn.Sequential(
                HighWayLayer(
                    n_ref_features
                ),
                nn.Linear(
                    n_ref_features,
                    self.embedding_size,
                    True
                ),
                nn.PReLU(),
                L2NormLayer(
                    axis=1,
                    eps=self.eps,
                ),
            )


            if not self.duplicate_neck:
                self.qry_neck = self.ref_neck

            else:
                self.qry_neck = nn.Sequential(
                    HighWayLayer(
                        n_ref_features
                    ),
                    nn.Linear(
                        n_qry_features,
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
            self.ref_neck = nn.Sequential(
                HighWayLayer(
                    n_ref_features
                ),
                nn.PReLU(),
                nn.Linear(
                    n_ref_features,
                    self.embedding_size,
                    True
                ),
                L2NormLayer(
                    axis=1,
                    eps=self.eps,
                ),
            )


            if not self.duplicate_neck:
                self.qry_neck = self.ref_neck

            else:
                self.qry_neck = nn.Sequential(
                    HighWayLayer(
                        n_ref_features
                    ),
                    nn.PReLU(),
                    nn.Linear(
                        n_qry_features,
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
            
        return None
    
    
    
    @torch.jit.ignore
    def predict_embedding(
            self,
            img,
            qry_neck=True,
            calc_triplet_cls=True
        ):

        if qry_neck:
            embed = self.forward_qry(img)
        else:
            embed = self.forward_ref(img)

        if not self.triplet_fw and calc_triplet_cls:
            projection = self.criterion.project(embed)

            return embed, projection
        else:
            return embed
            


# In[19]:


model = FacebookModel(args)


# In[20]:


# Number of Weights
# effnetv2m = 302.51M
# resnet50  = 275.44M


# In[21]:


# Last CheckPoint
last_ckpt = model.find_last_saved_ckpt()
if last_ckpt is not None:
    print(f' - Found ckpt: {last_ckpt}')


# In[22]:


if CPY_QRY2REF:
    print(' - Replacing REF Neck...')
    ckpt = torch.load(
        last_ckpt,
        map_location='cpu'
    )
    
    for k in ckpt['state_dict'].keys():
        if 'qry_neck' in k:
            ckpt['state_dict'][k.replace('qry_neck', 'ref_neck')] = ckpt['state_dict'][k].clone()
            
    last_ckpt = last_ckpt.replace('.ckpt', '_same_neck.ckpt')
    torch.save(
        ckpt,
        last_ckpt
    )


# In[23]:


# last_ckpt = './Sanjay/M2/FacebookModel_Eepoch=19_TLtrn_loss_epoch=16.8642_TAtrn_acc_epoch=0.0002_VLval_loss_epoch=17.0829_VAval_acc_epoch=0.2450_same_neck.ckpt'
# args.N_GPUS = 0
# args.precision = 32


# In[24]:


dl_trn = DataLoader(
        ds_trn,
        batch_size=args.BATCH_SIZE,
        num_workers=args.N_WORKERS,
        shuffle=True,
    )

dl_val = DataLoader(
    ds_val,
    batch_size=args.BATCH_SIZE,
    num_workers=args.N_WORKERS,
    shuffle=True,
)


# In[25]:


# for data in tqdm(dl_trn):
#     pass


# In[26]:


checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_acc_epoch',
    mode='max',
    save_last=True,
    save_top_k=5,
    dirpath=None,
    filename='FacebookModel_E{epoch:2d}_TL{trn_loss_epoch:0.04f}_TA{trn_acc_epoch:0.04f}_VL{val_loss_epoch:0.04f}_VA{val_acc_epoch:0.04f}'
)

if last_ckpt is not None:
    print(' - Restarting from:', last_ckpt)

trainer = pl.Trainer(
    default_root_dir=model.checkpoint_base_path,
    gradient_clip_val=model.clip_grad_norm,
    gradient_clip_algorithm='norm',
    limit_val_batches=0.05,
    gpus=args.N_GPUS,
    check_val_every_n_epoch=1,
    accumulate_grad_batches=args.n_steps_grad_update,
    precision=args.precision,
    max_epochs=1000,
    min_epochs=500,
    reload_dataloaders_every_n_epochs=1,
    auto_lr_find=False,
    callbacks=[checkpoint_callback],
    resume_from_checkpoint=last_ckpt,
    accelerator=args.accelerator if args.N_GPUS > 1 else None,
    auto_select_gpus=False,
    replace_sampler_ddp=True,
    num_sanity_val_steps=2,
)


# In[27]:


# N_params_to   = 0
# N_params_from = 456 # 258

# n_w = 0
# params_v = []
# params_names_v = []
# print('Optimizable parameters:')
# for i_p, (n, p) in enumerate( list( model.named_parameters() )):
#     if (i_p < N_params_to) or (i_p >= N_params_from):
#         p.requires_grad_(True)
#         params_v.append(p)
#         params_names_v.append(n)
    
#     else:
#         p.requires_grad_(False)
        
#     print('{:4d}  {:s}  {:40s}  {:}'.format(i_p, 'OPT' if p.requires_grad else '---',  n, p.shape) )
    
#     if p.requires_grad:
#         n_w += np.prod(p.shape)

# print(f'Total optimizable weights: {n_w/1e6:0.02f} Mw')


# In[28]:


trainer.fit(model, dl_trn, dl_val)


# In[29]:


sys.exit(0)
