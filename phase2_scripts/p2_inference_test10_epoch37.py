#!/usr/bin/env python
# coding: utf-8

# In[1]:


from model_inference import do_inference


# In[ ]:


from modules.Facebook_AFMultiGPU_model_v23_sy_v9 import ArgsT23_EffNetV2, FacebookModel
ckpt_filename = './checkpoints/sjy_test10/epoch=37-step=161955_LIGHT.ckpt'

args = ArgsT23_EffNetV2()

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

