#!/usr/bin/env python
# coding: utf-8

import os, sys, gc
sys.path.append('./modules')

import typer
from typing import Optional
from Facebook_AF_model_v27 import *

from model_config import *
from modules.eval_functions import *
from modules.eval_metrics import evaluate

def main(
    checkpoint_path:str='./TEST26_100k_nobias_l1/lightning_logs/version_0/checkpoints/last.ckpt',
    batch_size:int=400,
    ):

    print('Detected devices:' )
    for i in range( torch.cuda.device_count() ):
        print( f' *', str( torch.cuda.get_device_properties(i))[1:] )


    args                      = ArgsT26()
    args.BATCH_SIZE           = batch_size
    args.arc_classnum         = 40
    args.phase_2              = False
    print(args)


    K = 500
    faiss_gpu_id = 0
    do_simple_augmentation = False

    
    assert (args.DATASET_WH[0] >= args.OUTPUT_WH[0]) and (args.DATASET_WH[1] >= args.OUTPUT_WH[1])
    assert (args.arc_w_prop_v is None) or (len(args.arc_w_prop_v) == len(args.arc_devices_v)), f'KERNEL_GPUS and KERNEL_PROPS must have the same number of elements. len(KERNEL_GPUS)={len(KERNEL_GPUS)} != len(KERNEL_PROPS) {len(KERNEL_PROPS)}'
    assert len(all_screenshots_v) > 100, f'{len(all_screenshots_v)} were found in SCREENSHOT_DIR={SCREENSHOT_DIR}'


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
        public_gt = pd.read_csv( public_ground_truth_path )

    except:
        public_ground_truth_path = os.path.join(args.DS_INPUT_DIR, 'public_ground_truth.csv')
        public_gt = pd.read_csv( public_ground_truth_path )


    seed_everything(seed=args.seed, dt_same_seed=3600)


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



    # Creating model
    model = FacebookModel(args)
    _ = model.restore_checkpoint(checkpoint_path)


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


    print("Query Embeddings:")
    embed_qry_d = calc_embed_d(
        model, 
        dataloader=dl_qry_full,
        do_simple_augmentation=do_simple_augmentation,
    )

    print("Inference Embeddings:")
    embed_ref_d = calc_embed_d(
        model, 
        dataloader=dl_ref_full, 
        do_simple_augmentation=do_simple_augmentation
    )


    aug = '_AUG' if do_simple_augmentation else ''
    submission_path = checkpoint_path.replace('.ckpt', f'_{args.OUTPUT_WH[0]}x{args.OUTPUT_WH[1]}{aug}_REF.h5')

    save_submission(
        embed_qry_d,
        embed_ref_d,
        save_path=submission_path,
    )

    eval_d = evaluate(
        submission_path=submission_path,
        gt_path=public_ground_truth_path,
        is_matching=False,
    )


    return None

if __name__ == '__main__':
    typer.run(main)
