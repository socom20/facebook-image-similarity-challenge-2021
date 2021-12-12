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
import multiprocessing

import h5py
from tqdm import tqdm, tqdm_notebook

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
from PIL import Image

from torch.utils.data import Dataset, DataLoader

import shutil
import albumentations as A
import augly.image as imaugs
import augly.utils as utils


# # Aux Funtions

# In[2]:


def save_obj(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    print(f'Saved: {filename}')
    return None

def load_obj(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    print(f'Loaded: {filename}')
    return obj


# In[3]:


def get_path_from_sample_id(sample_id, ds_dir='./dataset', ext='jpg'):
    if sample_id[0] == 'Q':
        image_path = os.path.join(ds_dir, 'query_images', f'{sample_id}.{ext}')
    elif sample_id[0] == 'R':
        image_path = os.path.join(ds_dir, 'reference_images', f'{sample_id}.{ext}')
    elif sample_id[0] == 'T':
        image_path = os.path.join(ds_dir, 'training_images', f'{sample_id}.{ext}')
    elif sample_id[0] == 'I':
        image_path = os.path.join(ds_dir, 'imagenet_images', f'{sample_id}.{ext}')
    else:
        raise Exception('???')
        
    return image_path
    
def read_image(sample_id, ds_dir='./dataset', ext='jpg', use_cv2=False):
    image_path = get_path_from_sample_id(sample_id, ds_dir, ext)
    
    if use_cv2:
        img = cv2.imread(image_path)
        assert img is not None, f'Sample not found: {image_path}'
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    else:
        img = Image.open(image_path)
        
    return img


# In[4]:


# img = read_image('R022201').resize((512,512))
# metadata_v = []
# img = imaugs.color_jitter(
#             image=img,
#             brightness_factor=0.25 + 1.5 * random.random(),
#             contrast_factor=0.25 + 1.5 * random.random(),
#             saturation_factor=0.2 + 50*np.exp( 5*random.random() )/300.0, #0.2 + 0.5*np.exp( 10*random.random() )/300.0,
#             metadata=metadata_v,
#         )

# img


# # Image Rescaling

# In[5]:


def copy_task(in_samples_v, in_dir, out_dir, ext, output_wh, verbose=True):
    if verbose:
        from tqdm import tqdm_notebook
        itt = tqdm_notebook( in_samples_v )
    else:
        itt = in_samples_v
        
    for i_s, input_path in enumerate(itt):
        filename = os.path.basename( input_path )
        output_path = os.path.join( out_dir, filename.replace('jpg', ext) )
        
        img = Image.open(input_path)
        
        # Fix: Removing the alpha channel
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            img = img.convert('RGB')
        
        img = img.resize(output_wh)    
        img.save(output_path)
        
        if verbose and (i_s%10==0):
            itt.set_description(filename)
        
#         if i_s == 200:
#             itt.set_description(filename)
#             break

    return None


def resize_dataset(
    ds_input_dir=f'./dataset',
    ds_output_dir=None,
    output_wh=(512, 512),
    output_ext='jpg',
    num_workers=32,
    ALL_FOLDERS=['query_images', 'reference_images', 'training_images', 'imagenet_images'],
    verbose=True,
):
        
    for folder in ALL_FOLDERS:
        img_dir = os.path.join(ds_input_dir, folder)
        assert os.path.exists( img_dir ), f'ERROR, input dir = {img_dir} NotFound!!'
    
    
    if ds_input_dir[-1] in ['/', '\\']:
        ds_input_dir = ds_input_dir[:-1]
            
    if ds_output_dir is None:
        ds_output_dir = f'{ds_input_dir}_{output_ext}_{output_wh[0]}x{output_wh[1]}'

    for folder in ALL_FOLDERS:
        out_dir = os.path.join(ds_output_dir, folder)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            print(f' - Created: {out_dir}')
            
    shutil.copy(
        os.path.join(ds_input_dir, 'public_ground_truth.csv'),
        os.path.join(ds_output_dir, 'public_ground_truth.csv'),
    )            
            
    for folder in ALL_FOLDERS:
        in_dir = os.path.join(ds_input_dir, folder)
        out_dir = os.path.join(ds_output_dir, folder)

        all_in_samples_v = glob.glob( os.path.join(in_dir, '*.jpg') )
        all_out_samples_v = glob.glob( os.path.join(out_dir, '*.jpg') )
        
        if len(all_in_samples_v) == len(all_out_samples_v):
            print(f' - Skipping folder: "{folder}". The dataset, seems to be already created.')
            continue
        
        all_in_samples_v.sort()

        n_samples = len(all_in_samples_v)
        s_batch = max(1, n_samples//num_workers)

        p_v = []
        for i_w in range(num_workers):
            if i_w < num_workers-1:
                in_samples_v = all_in_samples_v[i_w*s_batch: (i_w+1)*s_batch]
            else:
                in_samples_v = all_in_samples_v[i_w*s_batch:]

            p = multiprocessing.Process(
                target=copy_task,
                args=(in_samples_v, in_dir, out_dir, output_ext, output_wh, verbose)
            )

            p_v.append(p)

        for p in p_v:
            p.start()

        for p in p_v:
            p.join()
            


# # Augmentation

# In[6]:


# bboxes_json_path = os.path.join(utils.ASSETS_BASE_DIR, 'screenshot_templates', 'bboxes.json') 
# all_screenshots_v = [
#     'mobile.png',
# #     'web.png'
# ]
# all_screenshots_v = [os.path.join(utils.ASSETS_BASE_DIR, 'screenshot_templates', filename) for filename in all_screenshots_v]


SCREENSHOT_DIR = './FB_page_qry'
bboxes_json_path = os.path.join(SCREENSHOT_DIR, 'bboxes.json')

with open(bboxes_json_path, 'r') as f:
    bboxes_d = json.load(f)

    
all_screenshots_v = [os.path.join(SCREENSHOT_DIR, fn) for fn in bboxes_d.keys() if os.path.exists(os.path.join(SCREENSHOT_DIR, fn))]

print(f' - Found: {len(all_screenshots_v)} screenshots.  SCREENSHOT_DIR={SCREENSHOT_DIR}')


# In[7]:


all_emojis_v = glob.glob(utils.EMOJI_DIR + "/*/*.png")
all_fonts_v = glob.glob(utils.FONTS_DIR + "/*.ttf")

with open(utils.TEXT_DIR+"/letter_unicode_mapping.json", "r") as f:
    letter_unicode_mapping_d = json.load(f)
    
with open(utils.TEXT_DIR+"/fun_fonts.json", "r") as f:
    fun_fonts_d = json.load(f)
    
with open(utils.TEXT_DIR+"/misspelling.json", "r") as f:
    misspelling_d = json.load(f)
    


# In[8]:


# Fast Fonts < 75 ms
all_fonts_v = [
#     'Raleway-SemiBoldItalic.ttf',
    'Quicksand-Italic.ttf',
    'SolveigText.ttf',
    'Titillium-Regular.ttf',
    'SourceSansPro-Black.ttf',
    'Titillium-Thin.ttf',
    'SourceSansPro-Bold.ttf',
    'blackjack.ttf',
    'Allura-Regular.ttf',
#     'Raleway-BoldItalic.ttf',
#     'Raleway-Medium.ttf',
    'OstrichSans-Medium.ttf',
#     'Raleway-Light.ttf',
    'Titillium-BoldUpright.ttf',
    'OpenSans-LightItalic.ttf',
#     'Raleway-ExtraLight.ttf',
    'PlayfairDisplaySC-Bold.ttf',
    'PlayfairDisplay-Italic.ttf',
    'Ubuntu-C.ttf',
    'OstrichSans-Black.ttf',
    'Elsie-Black.ttf',
    'Titillium-ThinItalic.ttf',
#     'Raleway-BlackItalic.ttf',
    'Impact.ttf',
    'OpenSans-ExtraBoldItalic.ttf',
    'Titillium-SemiboldUpright.ttf',
    'Ubuntu-R.ttf',
    'GreatVibes-Regular.ttf',
    'Ubuntu-RI.ttf',
    'cac_champagne.ttf',
    'CaviarDreams.ttf',
    'LeagueSpartan-Bold.ttf',
    'SolveigBold-Italic.ttf',
    'Titillium-Light.ttf',
    'PlayfairDisplay-BlackItalic.ttf',
#     'DavysDingbats.ttf',
    'OpenSans-ExtraBold.ttf',
    'Raleway-MediumItalic.ttf',
    'SolveigDemiBold-Italic.ttf',
    'modernpics.ttf',
    'Raleway-ThinItalic.ttf',
    'OstrichSans-Light.ttf',
    'infini-italique.ttf',
#     'Sofia-Regular.ttf',
    'Ubuntu-L.ttf',
    'Ubuntu-BI.ttf',
    'Quicksand-LightItalic.ttf',
#     'Raleway-Bold.ttf',
    'KaushanScript-Regular.ttf',
    'Titillium-Semibold.ttf',
    'berkshireswash-regular.ttf',
    'MountainsofChristmas.ttf',
#     'TypeMyMusic_1.1.ttf',
    'SourceSansPro-Semibold.ttf',
    'OpenSans-Light.ttf',
    'CaviarDreams_BoldItalic.ttf',
    'heydings_controls.ttf',
    'SolveigBold.ttf',
    'SourceSansPro-BlackIt.ttf',
    'Quicksand_Dash.ttf',
    'OstrichSans-Heavy.ttf',
    'Raleway-LightItalic.ttf',
    'Chunkfive.ttf',
    'ElsieSwashCaps-Black.ttf',
    'PlayfairDisplaySC-Regular.ttf',
    'symbol-signs.ttf',
    'OpenSans-Regular.ttf',
    'SolveigDemiBold.ttf',
    'Raleway-Regular.ttf',
    'Raleway-ExtraBoldItalic.ttf',
    'DancingScript-Regular.ttf',
    'ostrich-regular.ttf',
    'SourceSansPro-ExtraLightIt.ttf',
#     'Quicksand-Bold.ttf',
    'WCSoldOutABta.ttf',
    'Kalocsai_Flowers.ttf',
    'Quicksand-BoldItalic.ttf',
    'Titillium-RegularItalic.ttf',
    'Lobster_1.3.ttf',
    'Windsong.ttf',
    'OstrichSansRounded-Medium.ttf',
    'Raleway-SemiBold.ttf',
    'Ubuntu-M.ttf',
    'PlayfairDisplay-Black.ttf',
    'infini-romain.ttf',
    'SourceSansPro-SemiboldIt.ttf',
    'PlayfairDisplay-Regular.ttf',
    'SolveigDisplay.ttf',
    'Raleway-Italic.ttf',
    'OpenSans-Bold.ttf',
    'OstrichSans-Bold.ttf',
    'PlayfairDisplaySC-BoldItalic.ttf',
    'OpenSans-SemiboldItalic.ttf',
    'Raleway-Thin.ttf',
    'Titillium-Black.ttf',
    'SolveigDisplay-Italic.ttf',
    'PlayfairDisplaySC-BlackItalic.ttf',
    'SourceSansPro-It.ttf',
    'SourceSansPro-ExtraLight.ttf',
    'Titillium-SemiboldItalic.ttf',
    'PlayfairDisplay-BoldItalic.ttf',
    'ElsieSwashCaps-Regular.ttf',
    'SourceSansPro-BoldIt.ttf',
    'PlayfairDisplaySC-Italic.ttf',
    'SolveigText-Italic.ttf',
    'Titillium-LightItalic.ttf',
    'PlayfairDisplaySC-Black.ttf',
#     'Raleway-ExtraBold.ttf',
    'Titillium-LightUpright.ttf',
    'Ubuntu-B.ttf',
    'OpenSans-BoldItalic.ttf',
    'SourceSansPro-Regular.ttf',
    'Entypo.ttf',
    'Caviar_Dreams_Bold.ttf',
    'infini-picto.ttf',
    'CaviarDreams_Italic.ttf',
    'AlexBrush-Regular.ttf',
    'Raleway-ExtraLightItalic.ttf',
    'PlayfairDisplay-Bold.ttf',
    'heydings_icons.ttf',
    'SirucaPictograms1.1_.ttf',
    'Ubuntu-MI.ttf',
    'Titillium-Bold.ttf',
    'Titillium-RegularUpright.ttf',
    'Titillium-BoldItalic.ttf',
    'OpenSans-Semibold.ttf',
    'SourceSansPro-LightIt.ttf',
    'Pacifico.ttf',
    'Ubuntu-LI.ttf']


all_fonts_v = [os.path.join( utils.FONTS_DIR, f) for f in all_fonts_v]


# In[9]:


class ColorAugs():
    def invert_channel(
        self,
        image,
        invert_r=False,
        invert_g=False,
        invert_b=False,
        metadata=None,
    ):
        
        if len(image.shape) == 3 and image.shape[-1] == 3:
            if invert_r:
                image[:,:,0] = 255 - image[:,:,0]
                
            if invert_g:
                image[:,:,1] = 255 - image[:,:,1]
                
            if invert_b:
                image[:,:,2] = 255 - image[:,:,2]
            
            if metadata is not None:
                H,W = image.shape[:2]
                metadata.append(
                    {
                        'name':'ColorAugs.invert_channel',
                        'src_width': W,
                        'src_height': H,
                        'dst_width': W,
                        'dst_height': H,

                        'invert_r': invert_r,
                        'invert_g': invert_g,
                        'invert_b': invert_b,
                    }
                )
            
        return image
    
    
    def swap_channels(
        self,
        image,
        new_channel_order_v=[2,1,0],
        metadata=None,
    ):
        if len(image.shape) == 3 and image.shape[-1] == 3:
            image = image[:,:,new_channel_order_v]
            
            if metadata is not None:
                H,W = image.shape[:2]
                metadata.append(
                    {
                        'name':'ColorAugs.swap_channels',
                        'src_width': W,
                        'src_height': H,
                        'dst_width': W,
                        'dst_height': H,
                        'new_channel_order_v': new_channel_order_v,
                    }
                )
                
        return image
                
                
    def shift_channels(
        self,
        image,
        shift_v=[10, -10],
        metadata=None,
    ):
        if len(image.shape) == 3 and image.shape[-1] == 3:
            H_i,W_i,C_i = image.shape
            I, J = shift_v
            
            image[:,:,0] = np.roll(image[:,:,0], (I, J), axis=(0,1) )
            image[:,:,2] = np.roll(image[:,:,2], (-I, -J), axis=(0,1) )
            
            I, J = abs(I), abs(J)
            if I>0 and J>0:
                image = image[I:-I,J:-J] 
            elif I==0 and J>0:
                image = image[:,J:-J] 
            elif I>0 and J==0:
                image = image[I:-I] 
            
            
            if metadata is not None:
                H,W,C = image.shape
                metadata.append(
                    {
                        'name':'ColorAugs.shift_channels',
                        'src_width': W_i,
                        'src_height': H_i,
                        'dst_width': W,
                        'dst_height': H,
                        'shift_v': shift_v,
                    }
                )

        return image
    
    
    def to_np_array(
        self,
        image,
        metadata=None,
    ):
        image = np.array(image)
        
        if metadata is not None:
            H,W = image.shape[:2]
            metadata.append(
                {
                    'name':'ColorAugs.to_np_array',
                    'src_width': W,
                    'src_height': H,
                    'dst_width': W,
                    'dst_height': H,
                }
            )
        
        return image
    
    
    def resize(
        self,
        image,
        width=512,
        height=512,
        metadata=None,
    ):
        H_i,W_i = image.shape[:2]
        
        image = cv2.resize(
            image,
            (width, height),
        )
        
        if metadata is not None:
            H,W = image.shape[:2]
            
            metadata.append(
                {
                    'name':'ColorAugs.resize',
                    'src_width': W_i,
                    'src_height': H_i,
                    'dst_width': W,
                    'dst_height': H,

                    'width': width,
                    'height': height,
                }
            )
        
        return image
        
CA = ColorAugs()


# In[10]:


def query_augmentation(
    img,
    output_wh,
    qry_wh_info_d=None,
    ds_dir='./dataset',
    overlay_img_ids_v=None,
):
    global metadata_v
    
    metadata_v = []

#     if img.size != output_wh:
#         img = img.resize(output_wh)
        
        
    if random.random() <= 0.10:
        img = imaugs.scale(
            image=img,
            factor=0.5 + 0.4 * random.random(),
            interpolation=None,
            metadata=metadata_v
        )

    if random.random() <= 0.10:
        img = imaugs.rotate(
            image=img, 
            degrees=random.randint(0, 360),
            metadata=metadata_v
        )


    if random.random() <= 0.10:
        try:
            img = imaugs.perspective_transform(
                image=img, 
                sigma=30.0,
                dx=0.0,
                dy=0.0,
                seed=random.randint(0,1024),
                crop_out_black_border=random.choice([True,False]),
                metadata=metadata_v
            )
            
        except Exception as e:
            img = imaugs.perspective_transform(
                image=img, 
                sigma=30.0,
                dx=0.0,
                dy=0.0,
                seed=random.randint(0,1024),
                crop_out_black_border=False,
                metadata=metadata_v
            )
            

    if random.random() <= 0.10:
        img = imaugs.encoding_quality(
            image=img, 
            quality=random.randint(10, 50),
            metadata=metadata_v
        )

    if random.random() <= 0.10:
        img = imaugs.crop(
            image=img,
            x1= random.choice([0.0, 0.0625, 0.125, 0.25]),
            y1= random.choice([0.0, 0.0625, 0.125, 0.25]),
            x2= random.choice([0.75, 0.875, 0.9375, 1.0]),
            y2= random.choice([0.75, 0.875, 0.9375, 1.0]),
            metadata=metadata_v
            )

    if random.random() <= 0.25:
        img = imaugs.color_jitter(
            image=img,
            brightness_factor=0.25 + 1.5 * random.random(),
            contrast_factor=0.25 + 1.5 * random.random(),
            saturation_factor=0.2 + 50*np.exp( 5*random.random() )/300.0,
            metadata=metadata_v,
        )

    if random.random() <= 0.10:
        img = imaugs.grayscale(
            image=img,
            mode='luminosity',
            metadata=metadata_v,
        )

    if random.random() <= 0.05:
        img = imaugs.opacity(
            image=img,
            level=0.6 + 0.4 * random.random(),
            metadata=metadata_v
        )

    if random.random() <= 0.05:
        W, H = img.size
        if random.random() <= 0.5 or W == H:
            img = imaugs.pad(
                image=img,
                w_factor=0.10*random.random(),
                h_factor=0.10*random.random(),
                color=(random.randint(10,255), random.randint(10,255), random.randint(10,255)),
                metadata=metadata_v
            )

        else:
            img = imaugs.pad_square(
                image=img,
                color=(random.randint(10,255), random.randint(10,255), random.randint(10,255)),
                metadata=metadata_v
            )

    if random.random() <= 0.10:
        if overlay_img_ids_v is None:
            neg_img = read_image(f'T{random.randint(0,1000000):06d}', ds_dir=ds_dir)
            
        else:
            neg_img = read_image(random.choice(overlay_img_ids_v), ds_dir=ds_dir)
        
        if random.random() <= 0.50:
            img = imaugs.overlay_image(
                image=img,
                overlay=neg_img,
                opacity=min(1.0,0.2+random.random()),
                overlay_size=0.1 + 0.6*random.random(),
                x_pos=random.random(),
                y_pos=random.random(),
                metadata=metadata_v
            )
            
        else:
            overlay_size = 0.30 + 0.4*random.random()
            img = imaugs.overlay_image(
                image=neg_img,
                overlay=img,
                opacity=min(1.0,0.4+random.random()),
                overlay_size=overlay_size,
                x_pos=(1.0-overlay_size)*random.random(),
                y_pos=(1.0-overlay_size)*random.random(),
                metadata=metadata_v,
            )
            
            metadata_v[-1]['neg_img'] = neg_img

    if random.random() <= 0.10:
        if random.random() <= 0.50:
            img = imaugs.blur(
                img,
                radius=0.1 + random.randint(1,4),
                metadata=metadata_v
            )
            
        else:

            img = imaugs.sharpen(
                image=img, 
                factor=1.0 + 50*random.random(),
                metadata=metadata_v
            )

    if random.random() <= 0.10:
        img = imaugs.shuffle_pixels(
            image=img,
            factor=0.20*random.random(),
            seed=random.randint(0,1024),
            metadata=metadata_v,
        )

    if random.random() <= 0.10:
        img = imaugs.overlay_stripes(
            image=img,
            line_width=0.05 + 0.20*random.random(),
            line_color=(random.randint(10,255), random.randint(10,255), random.randint(10,255)),
            line_angle=180*random.random(),
            line_density=0.8*random.random(),
            line_type=random.choice(['solid', 'dotted', 'dashed']),
            line_opacity=min(1.0, 0.1 + random.random()),
            metadata=metadata_v
        )


    if random.random() <= 0.10:
        img = imaugs.overlay_emoji(
            image=img,
            emoji_path=random.choice(all_emojis_v),
            opacity=min(1.0,0.2+random.random()),
            emoji_size=min(1.0,0.05+random.random()),
            x_pos=random.random(),
            y_pos=random.random(),
            metadata=metadata_v,
        )

    if random.random() <= 0.10:
        font_file = random.choice(all_fonts_v)
        text = [random.randint(0, 2000) for _ in range(random.randint(5,10))]
        font_size = random.choice([0.10, 0.15, 0.25, 0.30, 0.40])
        try:
            img = imaugs.overlay_text(
                image=img,
                text=text,
                font_file=font_file,
                font_size=font_size,
                opacity=min(1.0,0.2+random.random()),
                color=(random.randint(10,255), random.randint(10,255), random.randint(10,255)),
                x_pos=0.25*random.random(),
                y_pos=0.75*random.random(),
                metadata=metadata_v
            )
            
        except Exception as e:
            raise Exception( f'Augly, overlay_text ERROR: font_file="{font_file}", font_size="{font_size}", text="{text}", Original Exception="{str(e)}"' )


    if random.random() <= 0.10:
        img = imaugs.pixelization(
            image=img,
            ratio=random.choice([0.125, 0.25, 0.5, 0.75]),
            metadata=metadata_v,
        )

    if random.random() <= 0.50:
        img = imaugs.hflip(
            image=img,
            metadata=metadata_v
        )

    if random.random() <= 0.10:
        img = imaugs.vflip(
            image=img,
            metadata=metadata_v
        )

    if random.random() <= 0.10:
        text = ''.join( random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=random.randint(2, 20)) )
        font_file = random.choice(all_fonts_v)
        caption_height = random.choice(
            ( 20,  25,  30,  35,  40,  45,  50,  55,  60,  65,  70,  75,  80,
              85,  90,  95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145,
             150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210,
             215, 220, 225, 230, 235, 240, 245))
        try:
            img = imaugs.meme_format(
                image=img,
                text=text,
                font_file=font_file,
                opacity=random.choice([0.5, 0.75, 1.0]),
                text_color=(random.randint(10,255), random.randint(10,255), random.randint(10,255)),
                caption_height=caption_height,
                meme_bg_color=(random.randint(10,255), random.randint(10,255), random.randint(10,255)),
                metadata=metadata_v,
            )
            
        except Exception as e:
            raise Exception( f'Augly, meme_format ERROR: text="{text}", font_file="{font_file}", caption_height="{caption_height}", Original Exception="{str(e)}"')
            

    if random.random() <= 0.05:
        img = imaugs.overlay_onto_screenshot(
            image=img,
            template_filepath=random.choice(all_screenshots_v),
            template_bboxes_filepath=bboxes_json_path,
            max_image_size_pixels=None,
            crop_src_to_fit=random.choice([True, False]),
            metadata=metadata_v
        )
        
    
    if qry_wh_info_d is not None:
        W, H = sample_wh(qry_wh_info_d)
        img = imaugs.resize(
            img,
            width=W,
            height=H,
            metadata=metadata_v
        )
    
    
#     if img.size != output_wh:
#         img = imaugs.resize(
#             img,
#             width=output_wh[0],
#             height=output_wh[1],
#             metadata=metadata_v
#         )

    img = CA.to_np_array(
        img,
        metadata=metadata_v
    )

    if random.random() <= 0.10:
        img = CA.invert_channel(
            img,
            invert_r=random.choice([True, False]),
            invert_g=random.choice([True, False]),
            invert_b=random.choice([True, False]),
            metadata=metadata_v
        )

    if random.random() <= 0.10:
        if random.random() <= 0.50:
            img = CA.swap_channels(
                img,
                new_channel_order_v=random.choices([0,1,2], k=3),
                metadata=metadata_v
            )
        else:
            new_channel_order_v = [0,1,2]
            random.shuffle(new_channel_order_v)
            img = CA.swap_channels(
                img,
                new_channel_order_v=new_channel_order_v,
                metadata=metadata_v
            )

    if random.random() <= 0.10:
        img = CA.shift_channels(
            img,
            shift_v=[random.randint(-25,25), random.randint(-25,25)],
            metadata=metadata_v
        )

    
    if img.shape[:2] != output_wh[::-1]:
        img = CA.resize(
            img,
            width=output_wh[0],
            height=output_wh[1],metadata=metadata_v
        )

    return img, metadata_v


def repeat_augmentation(img, metadata_v, resize_input_img=True):

    if len(metadata_v) > 0:
        if resize_input_img and (img.size) != (metadata_v[0]['src_width'], metadata_v[0]['src_height']):
            img = imaugs.resize(
                img,
                width=metadata_v[0]['src_width'],
                height=metadata_v[0]['src_height'],
            )
        
    for metadata_d in metadata_v:
        name = metadata_d['name']
#         print(name)
        if 'ColorAugs.' in name:
            aug_fun = getattr(CA, name.split('.')[1])
        else:
            aug_fun = getattr(imaugs, name)
            
        args_d = {k:tuple(v) if type(v) is list else v  for k,v in metadata_d.items() if k not in ['name', 'src_width', 'src_height', 'dst_width', 'dst_height', 'intensity']}
        
        if name == 'overlay_image' and 'neg_img' in args_d.keys():
            args_d['overlay'] = img
            img = args_d['neg_img']
            del(args_d['neg_img'])
            
        img = aug_fun(img, **args_d)
    
    return img


def reference_augmentation(img, output_wh=(512, 512) ):
    org_W, org_H = img.size
    metadata_v = []
    
#     if random.random() <= 0.10:
#         img = imaugs.rotate(
#             image=img, 
#             degrees=random.randint(-15, 15),
#             metadata=metadata_v
#         )
    
    if random.random() <= 0.10:
        img = imaugs.encoding_quality(
            image=img, 
            quality=random.randint(40, 100),
            metadata=metadata_v
        )
        
    if random.random() <= 0.50:
        img = imaugs.hflip(
            image=img,
            metadata=metadata_v
        )
    
    if img.size != output_wh:
        img = imaugs.resize(
                img,
                width=output_wh[0],
                height=output_wh[1],
                metadata=metadata_v

            )
    
    return img, metadata_v


# # DataSets

# In[11]:


def image_normalization(img, norm_type='simple'):
    if len(img.shape) != 3 or img.shape[-1] != 3:
        if len(img.shape) == 2:
            img = np.repeat( img[:,:,None], 3, axis=2)

        elif len(img.shape) == 3:
            if img.shape[-1] == 1:
                img = np.repeat( img, 3, axis=2)

            elif img.shape[-1] > 3:
                img = img[:,:,:3]

            else:
                raise Exception(f'Bad image shape = {img.shape}')

        else:
            raise Exception(f'Bad image shape = {img.shape}')

    
    if norm_type == 'simple':
        img = img / 255
        
    elif norm_type == 'imagenet':
        mean = np.array([123.675, 116.28 , 103.53 ], dtype=np.float32)
        std = np.array([58.395   , 57.120, 57.375   ], dtype=np.float32)
        img = img - mean
        img *= np.reciprocal(std, dtype=np.float32)
        
    elif norm_type == 'centered':
        img = img/127.5 - 1.0
        
    else:
        raise NotImplementedError(f'norm_type={norm_type}, allowed=[simple, imagenet, centered]')
        
    return img


# In[12]:


class FacebookSyntheticDataset(Dataset):
    def __init__(
        self,
        ref_samples_id_v,
        neg_samples_id_v=None,
        do_ref_augmentation=True,
        ds_dir='./dataset',
        output_wh=(512,512),
        max_retries=10,
        channel_first=True,
        return_neg_img=True,
        overlay_img_ids_v=None,
        norm_type='simple',
        verbose=True,
    ):
        
        self.ref_samples_id_v = ref_samples_id_v
        self.do_ref_augmentation = do_ref_augmentation
        self.neg_samples_id_v = neg_samples_id_v
        self.ds_dir = ds_dir
        self.output_wh = output_wh
        self.max_retries = max_retries
        self.channel_first = channel_first
        self.return_neg_img = return_neg_img
        self.overlay_img_ids_v = overlay_img_ids_v
        self.norm_type = norm_type
        self.verbose = verbose
        
        
        self.output_shape = (output_wh[1], output_wh[0], 3)
        
        
        self.A_ref = A.Cutout(
            num_holes=int(0.05 * np.prod(output_wh) / (min(output_wh)//20)**2),
            max_h_size=min(output_wh)//20,
            max_w_size=min(output_wh)//20,
            fill_value=0.0,
            p=0.5,
        )
        
        if self.neg_samples_id_v is not None:
            assert len(self.neg_samples_id_v) == len(self.ref_samples_id_v), 'Length ERROR.'
        
        if self.return_neg_img:
            self.all_img_keys_v = ['ref_img', 'qry_img', 'neg_img']
            
        else:
            self.all_img_keys_v = ['ref_img', 'qry_img']
            
        return None
    
    def __len__(self):
        return len(self.ref_samples_id_v)
    
    def set_neg_samples(self, neg_samples_id_v):
        assert len(self.ref_samples_id_v) == len(neg_samples_id_v), 'Length ERROR.'
        self.neg_samples_id_v = neg_samples_id_v
        return None
    
    
    def __getitem__(self, idx):
        if idx < 0:
            idx += self.__len__()
        
        ref_sample_id = self.ref_samples_id_v[idx]
        
        i_try = 0
        aug_ok = False
        while not aug_ok:
            try:
                ref_img = read_image(
                    ref_sample_id,
                    ds_dir=self.ds_dir
                )
                
                qry_img, qry_metadata_v = query_augmentation(
                    ref_img,
                    output_wh=self.output_wh,
                    qry_wh_info_d=None,
                    ds_dir=self.ds_dir,
                    overlay_img_ids_v=self.overlay_img_ids_v,
                )
                qry_sample_id = ref_sample_id + '_SYN'
                aug_ok = True
            
            except KeyboardInterrupt as e:
                raise e
                
            except Exception as e:
                i_try += 1
                if i_try > self.max_retries:
                    raise e
                else:
                    print('WARNING, query_augmentation:', str(e))
        
        
        if self.do_ref_augmentation:
            ref_img, ref_metadata_v = reference_augmentation(
                img=ref_img,
                output_wh=self.output_wh
            )
            
            ref_img = np.array(ref_img, dtype=np.float32)
            ref_img = self.A_ref(image=ref_img)['image']
            
        else:
            if ref_img.size != self.output_wh:
                ref_img = imaugs.resize(
                    ref_img,
                    width=self.output_wh[0],
                    height=self.output_wh[1]
                )
        
            ref_img = np.array(ref_img, dtype=np.float32)
            
        ret_d = {
            'cls': idx,
            'ref_sample_id': ref_sample_id,
            'qry_sample_id': qry_sample_id,
            
            'ref_img': image_normalization(ref_img, norm_type=self.norm_type),
            'qry_img': image_normalization(qry_img, norm_type=self.norm_type),
        }
        
        
        if self.return_neg_img:
            if self.neg_samples_id_v is not None:
                neg_sample_id = random.choice(self.neg_samples_id_v[idx])

            else:
#                 while ( neg_sample_id := random.choice(self.ref_samples_id_v) ) == ref_sample_id: pass
                
                neg_sample_id = random.choice(self.ref_samples_id_v)
                while ( neg_sample_id == ref_sample_id):
                    neg_sample_id = random.choice(self.ref_samples_id_v)

            neg_img = read_image(
                neg_sample_id,
                ds_dir=self.ds_dir
            )

            neg_img = repeat_augmentation(
                neg_img,
                qry_metadata_v
            )
            
            ret_d['neg_sample_id'] =  neg_sample_id
            ret_d['neg_img']       =  image_normalization(np.array(neg_img, dtype=np.float32), norm_type=self.norm_type)

            
        if self.channel_first:
            for k in self.all_img_keys_v:
                ret_d[k] = ret_d[k].transpose( (2,0,1) )
        
        return ret_d
    

    def plot_sample(self, idx):
        def norm(img):
            img_min = img.min()
            print(f' ImShow normalization: img.min()={img.min():0.02f}  img.max()={img.max():0.02f}')
            if img_min < 0.0:
                img = img - img_min
            img_max = img.max()
            if img_max > 1.0:
                img = img / img_max
            
            return img
        
        ret_d = self[idx]
        
        f, axes = plt.subplots(1,3 if self.return_neg_img else 2, figsize=(15,5))
        
        if self.channel_first:
            img = ret_d['ref_img'].transpose( (1,2,0) )
        else:
            img = ret_d['ref_img']

        if self.norm_type != 'simple':
            img = norm(img)

        axes[0].imshow( img )
        axes[0].set_title('Ref_Img:' + ret_d['ref_sample_id'])

        if self.channel_first:
            img = ret_d['qry_img'].transpose( (1,2,0) )
        else:
            img = ret_d['qry_img']

        if self.norm_type != 'simple':
            img = norm(img)

        axes[1].imshow( img )
        axes[1].set_title('Qry_Img:' + ret_d['qry_sample_id'])
        if self.return_neg_img:
            if self.channel_first:
                img = ret_d['neg_img'].transpose( (1,2,0) )
            else:
                img = ret_d['neg_img']

            if self.norm_type != 'simple':
                img = norm(img)

            axes[2].imshow( img )
            axes[2].set_title('Neg_Img:' + ret_d['neg_sample_id'])

                
        plt.tight_layout()
        plt.show()
        
        return None

    
# ds = FacebookSyntheticDataset(
#     ref_samples_id_v=[f'T{i:06d}' for i in range(29)],
#     neg_samples_id_v=None,
# )

# ds.plot_sample(4)


# In[13]:


class FacebookDataset(Dataset):
    def __init__(
        self,
        samples_id_v,        
        do_augmentation=False,
        ds_dir='./dataset',
        output_wh=(512,512),
        channel_first=True,
        norm_type='simple',
        verbose=True,
    ):
        
        self.samples_id_v = samples_id_v
        self.do_augmentation = do_augmentation
        self.ds_dir = ds_dir
        self.output_wh = output_wh
        self.channel_first = channel_first
        self.norm_type = norm_type
        self.verbose = verbose
        
        
        self.output_shape = (output_wh[1], output_wh[0], 3)
        
        self.A_ref = A.Cutout(
            num_holes=int(0.05 * np.prod(output_wh) / (min(output_wh)//20)**2),
            max_h_size=min(output_wh)//20,
            max_w_size=min(output_wh)//20,
            fill_value=0.0,
            p=0.5,
        )
        
        return None
    
    
    def __len__(self):
        return len(self.samples_id_v)
    
    
    def __getitem__(self, idx):
        if idx < 0:
            idx += self.__len__()
        
        sample_id = self.samples_id_v[idx]

        img = read_image(
            sample_id,
            ds_dir=self.ds_dir
        )
        
        if self.do_augmentation:
            img, metadata_v = reference_augmentation(
                img=img,
                output_wh=self.output_wh
            )
            
            img = np.array(img, dtype=np.float32)
            img = self.A_ref(image=img)['image']
            
        else:
            if img.size != self.output_wh:
                img = imaugs.resize(
                    img,
                    width=self.output_wh[0],
                    height=self.output_wh[1]
                )
            
            img = np.array(img, dtype=np.float32)
            
        img = image_normalization(img, norm_type=self.norm_type)
        
        
        
        if self.channel_first:
            img = img.transpose( (2,0,1) )
            
        ret_d = {
            'sample_id': sample_id,
            'img': img,
            'cls': idx,
        }
        
        return ret_d
    

    def plot_sample(self, idx):
        def norm(img):
            print(f' ImShow normalization: img.min()={img.min():0.02f}  img.max()={img.max():0.02f}')
            img_min = img.min()
            if img_min < 0.0:
                img = img - img_min
            img_max = img.max()
            if img_max > 1.0:
                img = img / img_max
            
            return img
        
        ret_d = self[idx]
        
        f, axes = plt.subplots(1,1, figsize=(5,5))
        
        if self.channel_first:
            img = ret_d['img'].transpose( (1,2,0) )
        else:
            img = ret_d['img']
            
            
        if self.norm_type != 'simple':
            img = norm(img)
            
        axes.imshow( img )
        
        axes.set_title('Img:' + ret_d['sample_id'])
        plt.tight_layout()
        plt.show()
        
        return None


# In[14]:


if __name__ == '__main__':
    DS_INPUT_DIR = f'./dataset'  # Path where "query_images", "reference_images" and "training_images" are located

    # OUTPUT_WH = (224, 224)
    OUTPUT_WH = (384, 384)
#     OUTPUT_WH = (512, 512)
    
    DS_DIR = f'{DS_INPUT_DIR}_jpg_{OUTPUT_WH[0]}x{OUTPUT_WH[1]}'  # path where the rescaled images will be saved

    N_WORKERS  = 20


# In[15]:


if __name__ == '__main__':
    ALL_FOLDERS = ['query_images', 'reference_images', 'training_images', 'imagenet_images']
    
    if any( [not os.path.exists(os.path.join(DS_DIR, folder)) for folder in ALL_FOLDERS] ):
        resize_dataset(
            ds_input_dir=DS_INPUT_DIR,
            ds_output_dir=DS_DIR,
            output_wh=OUTPUT_WH,
            output_ext='jpg',
            num_workers=N_WORKERS,
            ALL_FOLDERS=ALL_FOLDERS,
            verbose=True,
        )


# In[16]:


if __name__ == '__main__':
    n_trn_samples = 50_000
    n_val_samples = 25_000
    
    # ref_samples_id_v = np.array([f'R{i:06d}' for i in range(0, 1_000_000-n_val_samples)])
    trn_samples_id_v = np.array([f'T{i:06d}' for i in range(0, n_trn_samples)])
    neg_samples_id_v = None

    ds_trn = FacebookSyntheticDataset(
        ref_samples_id_v=trn_samples_id_v,
        neg_samples_id_v=neg_samples_id_v,

        do_ref_augmentation=True,
        ds_dir=DS_DIR,
        output_wh=OUTPUT_WH,
        max_retries=10,
        channel_first=False,
        return_neg_img=False,
        norm_type='imagenet',
        verbose=True,
    )

    ds_trn.plot_sample(8)



    # ref_samples_id_v = np.array([f'R{i:06d}' for i in range(1_000_000-n_val_samples, 1_000_000)])
    ref_samples_id_v = np.array([f'R{i:06d}' for i in range(n_trn_samples, n_trn_samples+n_val_samples)])
    neg_samples_id_v = None

    ds_val = FacebookSyntheticDataset(
        ref_samples_id_v=ref_samples_id_v,
        neg_samples_id_v=neg_samples_id_v,

        do_ref_augmentation=True,
        ds_dir=DS_DIR,
        output_wh=OUTPUT_WH,
        max_retries=10,
        channel_first=True,
        return_neg_img=True,
        norm_type='imagenet',
        verbose=True,
    )

    ds_val.plot_sample(8)


# In[17]:


if __name__ == '__main__':
    ds_qry_full = FacebookDataset(
        samples_id_v=[f'Q{i:05d}' for i in range(25_000)],
        do_augmentation=False,
        ds_dir=DS_DIR,
        output_wh=OUTPUT_WH,
        channel_first=True,
        norm_type='imagenet',
        verbose=True,
    )
    ds_qry_full.plot_sample(4)


    ds_ref_full = FacebookDataset(
        samples_id_v=[f'R{i:06d}' for i in range(1_000_000)],
        do_augmentation=False,
        ds_dir=DS_DIR,
        output_wh=OUTPUT_WH,
        channel_first=True,
        norm_type='imagenet',
        verbose=True,
    )
    ds_ref_full.plot_sample(4)


    ds_trn_full = FacebookDataset(
        samples_id_v=[f'T{i:06d}' for i in range(1_000_000)],
        do_augmentation=True,
        ds_dir=DS_DIR,
        output_wh=OUTPUT_WH,
        channel_first=True,
        norm_type='imagenet',
        verbose=True,
    )
    ds_trn_full.plot_sample(4)


# In[18]:


# for i in tqdm( range(len(all_screenshots_v)) ):
#     img = read_image('I0004131')

#     metadata_v = []
#     img = imaugs.overlay_onto_screenshot(
#         image=img,
#         template_filepath=all_screenshots_v[i],
#         template_bboxes_filepath=bboxes_json_path,
#         max_image_size_pixels=None,
#         crop_src_to_fit=True,
#         metadata=metadata_v
#     )
#     display( img )


# In[19]:


# if __name__ == '__main__':
#     for data in tqdm(DataLoader(ds_trn, num_workers=1)):
#         pass

