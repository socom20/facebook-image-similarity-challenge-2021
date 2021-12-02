import os, sys
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
import cv2
from tqdm import tqdm


def uncompress_inplace(all_zip_v):

    itt = tqdm(all_zip_v)
    for file in itt:
        itt.set_description('Unzipping:' + os.path.basename(file))
        os.system(f'unzip "{file}"')

    return None        
    
def extract_frames(all_mp4_v, output_folder='./frames', n_frames_to_extract=3, output_format='F{:06d}.jpg'):

    if not os.path.exists(output_folder):
        print(' - Creating Output folder: ', output_folder)
        os.makedirs(output_folder)

    i_sample = 0
    itt = tqdm(all_mp4_v)
    for file in itt:
        itt.set_description('Reading:' + os.path.basename(file))
        cap = cv2.VideoCapture(file)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        i_s = int(frame_count*0.1)
        d_frame = (frame_count - 2*i_s) // n_frames_to_extract
        for i_f in range(i_s, frame_count, d_frame):
            cap.set(1, i_f)
            ret, frame = cap.read()
            while (frame == 0).all() and frame < frame_count:
                ret, frame = cap.read()
                
            cv2.imwrite(
                os.path.join(
                    output_folder,
                    output_format.format(i_sample)
                ),
                frame
            )

            i_sample += 1

        cap.release()
        

    return None  

if __name__ == '__main__':

    OUTPUT_FOLDER       = './face_frames'
    OUTPUT_FORMAT       = 'F{:06d}.jpg'
    N_FRAMES_TO_EXTRACT = 3

    
    all_zip_v = glob('./**/*.zip', recursive=True)
    print(f' - Found {len(all_zip_v)} zip files.')

    
    r = input(' |-> do you want to unzip them (y/n)? ')
    while r.lower() not in ['y', 'n']:
        r = input(' |-> do you want to unzip them (y/n)? '))

    if r == 'y':
        uncompress_inplace(all_zip_v)

    print()        
    
    all_mp4_v = glob('./**/*.mp4', recursive=True)
    print(f' - Found {len(all_mp4_v)} mp4 files.')


    r = input(' |-> do you want to extract frames (y/n)? ')
    while r.lower() not in ['y', 'n']:
        r = input(' |-> do you want to extract frames (y/n)? ')

    if r == 'y':
        extract_frames(
            all_mp4_v,
            output_folder=OUTPUT_FOLDER,
            n_frames_to_extract=N_FRAMES_TO_EXTRACT,
            output_format=OUTPUT_FORMAT,
            )
