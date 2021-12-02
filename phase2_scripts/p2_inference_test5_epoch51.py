#!/usr/bin/env python
# coding: utf-8

# In[1]:


from model_inference import do_inference


# In[ ]:


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


model = FacebookModel(args)
_ = model.restore_checkpoint(ckpt_filename)

do_inference(model, args, ckpt_filename)

