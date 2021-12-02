#!/usr/bin/env python
# coding: utf-8

# In[1]:


from model_inference import do_inference


# In[ ]:


from modules.Facebook_AF_model_v26 import ArgsT26, FacebookModel
ckpt_filename = './checkpoints/smp_test26/FacebookModel_epoch=27_trn_loss_epoch=0.7062_trn_acc_epoch=0.0000_val_loss_epoch=0.3296_val_acc_epoch=0.9935.ckpt'

args = ArgsT26()

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

