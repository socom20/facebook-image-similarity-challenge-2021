#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
from tqdm import tqdm
from glob import glob
from PIL import Image
import shutil


# In[2]:


INPUT_DIR  = './CLS-LOC' # where ImageNet's folders: 'test', 'train' and 'val' are located.
OUTPUT_DIR = '/home/sergio/kaggle' # The output directory were the folder "imagenet_images" will be created.
copy_files = True  # If False it moves the files


# In[3]:


PREFIX = 'I'
IMAGENET_FOLDER = 'imagenet_images'


# In[4]:


OUTPUT_DIR = os.path.join( OUTPUT_DIR, IMAGENET_FOLDER )


# In[5]:


all_folders_v = ['train', 'val', 'test']
for d in all_folders_v:
    assert d in os.listdir(INPUT_DIR), f'ImageNet data directory "{d}" not found.'


# In[6]:


def sort_paths(all_files_v):
    p_v = []
    for p in all_files_v:
        cls, num = os.path.basename(p).split('_')[-2:]
        cls = cls[1:]
        num = int(num[:-5])
        p_v.append( (cls, num, p) )
            
    p_v.sort()
    
    out_paths_v = [l[-1] for l in p_v]
    
    return out_paths_v


# In[7]:


all_files_v = []
for folder in tqdm(all_folders_v, desc='Searching ...'):
    files_v = glob(
        os.path.join(INPUT_DIR, folder, '**/*.JPEG'),
        recursive=True,
    )

    files_v = sort_paths(files_v)
    
    all_files_v.extend(files_v)

print(f' - n_images = {len(all_files_v)}')


# In[8]:


def get_next_file_number(path):
    dest_files_v = glob(
        os.path.join(path, '*.jpg'),
        recursive=True,
    )
    
    dest_files_v = [int( os.path.basename(p)[1:-4] )  for p in dest_files_v if len(p) > 5]
    
    if len(dest_files_v) > 0:
        return max(dest_files_v) + 1
    
    else:
        return 0


# In[9]:


os.makedirs(OUTPUT_DIR, exist_ok=True)

if copy_files:
    next_fn = 0
    
else:
    next_fn = get_next_file_number(OUTPUT_DIR)

for i_f, src_file in enumerate(tqdm(all_files_v, desc='CopyingFiles ...' if copy_files else 'MovingFiles ...')):
    dest_file = os.path.join(
        OUTPUT_DIR,
        f'{PREFIX}{next_fn + i_f:07d}.jpg'
    )
    
    if copy_files:
        shutil.copy(src_file, dest_file)
    else:
        shutil.move(src_file, dest_file)


# In[ ]:




