#!/usr/bin/env python
# coding: utf-8

# In[1]:


from model_inference import do_inference


# In[ ]:


from modules.Facebook_AF_model_v25 import ArgsT25_NfNetl1_ImageNet, FacebookModel
ckpt_filename = './checkpoints/smp_test25/FacebookModel_epoch=14_trn_loss_epoch=0.8997_trn_acc_epoch=0.0000_val_loss_epoch=0.4437_val_acc_epoch=0.9912.ckpt'

args = ArgsT25_NfNetl1_ImageNet()

args.BATCH_SIZE = 128
args.N_WORKERS = 7
args.DS_INPUT_DIR = f'./all_datasets/dataset'

args.pretrained_bb = False
args.arc_classnum = 40
args.ALL_FOLDERS = ['query_images', 'reference_images', 'training_images']

args.faiss_gpu_id = 0
args.phase_2 = True


model = FacebookModel(args)
_ = model.restore_checkpoint(ckpt_filename)

do_inference(model, args, ckpt_filename)

