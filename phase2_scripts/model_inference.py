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



from IPython.display import display, clear_output

import faiss
from modules.AugsDS_v13 import *
from modules.eval_functions import *
from modules.eval_metrics import evaluate

sys.path.append('./modules')


# In[2]:

def do_inference(model, args, ckpt_filename):
    # # Building model

    # In[3]:


    # # Inference configuration

    # In[5]:


    do_simple_augmentation = False
    K = 500
    faiss_gpu_id = args.faiss_gpu_id


    # In[6]:


    while args.DS_INPUT_DIR[-1] in ['/', r'\\']:
        args.DS_INPUT_DIR = args.DS_INPUT_DIR[:-1]

    # Path where the rescaled images will be saved
    args.DS_DIR = f'{args.DS_INPUT_DIR}_jpg_{args.DATASET_WH[0]}x{args.DATASET_WH[1]}'

    print(args)

    # # Data Source

    # In[7]:


    if any( [not os.path.exists(os.path.join(args.DS_DIR, folder)) for folder in args.ALL_FOLDERS] ):
        assert os.path.exists(args.DS_INPUT_DIR), f'DS_INPUT_DIR not found: {args.DS_INPUT_DIR}'

        resize_dataset(
            ds_input_dir=args.DS_INPUT_DIR,
            ds_output_dir=args.DS_DIR,
            output_wh=args.DATASET_WH,
            output_ext='jpg',
            num_workers=args.N_WORKERS,
            ALL_FOLDERS=args.ALL_FOLDERS,
            verbose=False,
        )

    print('Paths:')
    print(' - DS_INPUT_DIR:', args.DS_INPUT_DIR)
    print(' - DS_DIR:      ', args.DS_DIR)

    assert os.path.exists(args.DS_DIR), f'DS_DIR not found: {args.DS_DIR}'

    try:
        public_ground_truth_path = os.path.join(args.DS_DIR, 'public_ground_truth.csv')
        public_gt = pd.read_csv( public_ground_truth_path)

    except:
        public_ground_truth_path = os.path.join(args.DS_INPUT_DIR, 'public_ground_truth.csv')
        public_gt = pd.read_csv( public_ground_truth_path)


    # # Datasets

    # In[8]:


    ds_qry_full = FacebookDataset(
        samples_id_v=[f'Q{i:05d}' for i in (range(50_000, 100_000) if args.phase_2 else range(0, 50_000))] ,
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



    dl_qry_full = DataLoader(
            ds_qry_full,
            batch_size=args.BATCH_SIZE,
            num_workers=args.N_WORKERS,
            shuffle=False,
        )

    dl_ref_full = DataLoader(
        ds_ref_full,
        batch_size=args.BATCH_SIZE,
        num_workers=args.N_WORKERS,
        shuffle=False,
    )

    dl_trn_full = DataLoader(
        ds_trn_full,
        batch_size=args.BATCH_SIZE,
        num_workers=args.N_WORKERS,
        shuffle=False,
    )


    # In[9]:


    aug = '_AUG' if do_simple_augmentation else ''
    submission_path = ckpt_filename.replace('.ckpt', f'_{args.OUTPUT_WH[0]}x{args.OUTPUT_WH[1]}{aug}_REF.h5')
    scores_path = submission_path.replace('.h5', '_match_d.pickle')


    # ### Query embeddings

    # In[10]:


    embed_qry_d = calc_embed_d(
        model, 
        dataloader=dl_qry_full,
        do_simple_augmentation=do_simple_augmentation,
    )


    # ### Reference embeddings

    # In[12]:


    if not os.path.exists(submission_path):
        embed_ref_d = calc_embed_d(
            model, 
            dataloader=dl_ref_full, 
            do_simple_augmentation=do_simple_augmentation
        )

    else:
        _, embed_ref_d = read_submission(submission_path)

        
    save_submission(
        embed_qry_d,
        embed_ref_d,
        save_path=submission_path,
    )
        
    match_d = calc_match_scores(embed_qry_d, embed_ref_d, k=K, gpu_id=faiss_gpu_id)
    save_obj(match_d, scores_path)


    # ### Public GT validation

    # In[16]:


    if not args.phase_2:
        eval_d = evaluate(
            submission_path=submission_path,
            gt_path=public_ground_truth_path,
            is_matching=False,
        )


    # ### Training embeddings

    # In[17]:


    aug = '_AUG' if do_simple_augmentation else ''
    submission_path = ckpt_filename.replace('.ckpt', f'_{args.OUTPUT_WH[0]}x{args.OUTPUT_WH[1]}{aug}_TRN.h5')
    scores_path = submission_path.replace('.h5', '_match_d.pickle')


    # In[13]:


    if not os.path.exists(submission_path):
        embed_trn_d = calc_embed_d(
            model, 
            dataloader=dl_trn_full, 
            do_simple_augmentation=do_simple_augmentation
        )
        
    else:
        _, embed_trn_d = read_submission(submission_path)
        
        
    save_submission(
        embed_qry_d,
        embed_trn_d,
        save_path=submission_path,
    )


    # In[14]:


    match_d = calc_match_scores(embed_qry_d, embed_trn_d, k=K, gpu_id=faiss_gpu_id)
    save_obj(match_d, scores_path)




if __name__ == '__main__':
    from modules.Facebook_model_v20 import ArgsT15_EffNetV2L, FacebookModel

    ckpt_filename = './checkpoints/sjy_test5/FacebookModel_Eepoch=51_TLtrn_loss_epoch=0.8669_TAtrn_acc_epoch=0.9914_VLval_loss_epoch=0.4390_VAval_acc_epoch=0.9930.ckpt'

    args = ArgsT15_EffNetV2L()

    args.BATCH_SIZE = 64
    args.N_WORKERS = 7
    args.DS_INPUT_DIR = f'./all_datasets/dataset'

    args.pretrained_bb = False
    args.arc_classnum = 40
    args.ALL_FOLDERS = ['query_images', 'reference_images', 'training_images']

    args.faiss_gpu_id = 0
    args.phase_2 = True

    print(args)


    model = FacebookModel(args)
    _ = model.restore_checkpoint(ckpt_filename)

    do_inference(model, args, ckpt_filename)
