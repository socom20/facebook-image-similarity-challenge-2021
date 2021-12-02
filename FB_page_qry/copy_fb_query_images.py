import os, sys
import shutil



DS_INPUT_DIR = '../all_datasets/dataset/query_images'
OUTPUT_DIR = './'
images_file = './FB_queries.txt'


if __name__ == '__main__':
    with open(images_file, 'r') as f:
        qry_num_v = [int(l) if l[-1] != '\n' else int(l) for l in f.readlines() if len(l)>1]


    for i_s in qry_num_v:
        shutil.copy(
            os.path.join(DS_INPUT_DIR, f'Q{i_s:05d}.jpg'),
            os.path.join(OUTPUT_DIR, f'Q{i_s:05d}.jpg'),
            )
