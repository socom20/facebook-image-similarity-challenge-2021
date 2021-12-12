#!/usr/bin/env python
# coding: utf-8

import os, sys, gc
sys.path.append('./modules')

import typer
from typing import Optional
from Facebook_AFMultiGPU_model_v23_sy_v9 import *


def main(
    n_facebook_samples:int=5_000,  # 1_000_000 Facebook's training dataset
    n_imagenet_samples:int=0,       # 1_431_167 ImageNet
    batch_size:int=32*5,
    qry_augs_intensity:float=0.75,
    build_w_matrix:Optional[str]=None,
    show_some_samples:bool=False,
    ):

    print('Detected devices:' )
    for i in range( torch.cuda.device_count() ):
        print( f' *', str( torch.cuda.get_device_properties(i))[1:] )


    args                      = ArgsT23_EffNetV2()
    args.BATCH_SIZE           = batch_size
    args.start_ckpt           = build_w_matrix
    args.build_w_matrix       = build_w_matrix is not None
    args.QRY_AUGS_INTENSITY   = qry_augs_intensity

    args.n_facebook_samples   = n_facebook_samples
    args.n_imagenet_samples   = n_imagenet_samples
    args.arc_classnum         = n_facebook_samples + n_imagenet_samples
    args.checkpoint_base_path = f'./RUN_{int(args.arc_classnum/1000)}k_samples'
    print(args)

    
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
        public_gt = pd.read_csv( os.path.join(args.DS_DIR, 'public_ground_truth.csv') )

    except:
        public_gt = pd.read_csv( os.path.join(args.DS_INPUT_DIR, 'public_ground_truth.csv') )


    seed_everything(seed=args.seed, dt_same_seed=3600)



    ref_samples_id_v  = []
    overlay_img_ids_v = []
    
    if args.n_facebook_samples > 0:
        Facebook_samples_v = np.array( [f'T{i:06d}' for i in range(1_000_000)] )
        assert args.n_facebook_samples <= 1_000_000
        
        ref_samples_id_v.extend( Facebook_samples_v[:args.n_facebook_samples])
        overlay_img_ids_v.extend(Facebook_samples_v[args.n_facebook_samples:])
    
    #if args.n_imagenet_samples > 0:
    ImageNet_samples_v = load_obj(args.ImgNet_SAMPLES)
    assert args.n_imagenet_samples <= len(ImageNet_samples_v)

    ref_samples_id_v.extend( ImageNet_samples_v[:args.n_imagenet_samples])
    overlay_img_ids_v.extend(ImageNet_samples_v[args.n_imagenet_samples:])
        
    if args.n_frm_face_samples > 0:
        FrmFaces_samples_v = load_obj(args.FrmFaces_SAMPLES)
        assert args.n_frm_face_samples <= len(FrmFaces_samples_v)
        
        ref_samples_id_v.extend( FrmFaces_samples_v[:args.n_frm_face_samples])

    ref_samples_id_v = np.array(ref_samples_id_v)
    overlay_img_ids_v = np.array(overlay_img_ids_v)

    assert args.arc_classnum == len(ref_samples_id_v)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


    ds_trn = FacebookSyntheticDataset(
        ref_samples_id_v=ref_samples_id_v.copy(),
        neg_samples_id_v=None,

        do_ref_augmentation=False,
        ds_dir=args.DS_DIR,
        output_wh=args.OUTPUT_WH,
        max_retries=10,
        channel_first=True,
        return_neg_img=('arcface' not in args.criterion_name.lower()),
        overlay_img_ids_v=overlay_img_ids_v.copy(),
        norm_type=args.img_norm_type,
        qry_aug_intensity=args.QRY_AUGS_INTENSITY,
        verbose=True,
    )

    if show_some_samples:
        for i in range(10):
            ds_trn.plot_sample(i)


    ds_val = FacebookSyntheticDataset(
        ref_samples_id_v=ref_samples_id_v.copy(),
        neg_samples_id_v=None,

        do_ref_augmentation=False,
        ds_dir=args.DS_DIR,
        output_wh=args.OUTPUT_WH,
        max_retries=10,
        channel_first=True,
        return_neg_img=('arcface' not in args.criterion_name.lower()),
        overlay_img_ids_v=overlay_img_ids_v.copy(),
        norm_type=args.img_norm_type,
        qry_aug_intensity=args.QRY_AUGS_INTENSITY,
        verbose=True,
    )

    if show_some_samples:
        for i in range(10):
            ds_val.plot_sample(i)


    # Creating model
    model = FacebookModel(args)

    
    # Building ArcFace centroids matrix    
    if args.build_w_matrix:
        W_path = os.path.join(args.checkpoint_base_path, 'W.pickle')
        n_qry_evals = 4
        
        if (not os.path.exists(W_path)):
            print(' Creating W matrix ...')

            # Embedings from reference images
            W = calc_w_matrix(model, ds_trn, batch_size=64, num_workers=15, device='cuda:0', img_key='ref_img')

            # Embedings from synthetic query images
            for _ in range(n_qry_evals):
                W += calc_w_matrix(model, ds_trn, batch_size=64, num_workers=15, device='cuda:0', img_key='qry_img')

            W = W / (n_qry_evals+1)

            # Saving W matrix
            os.makedirs(args.checkpoint_base_path, exist_ok=True)
            save_obj(W, W_path)
        
        # Loading W matrix
        W = load_obj( W_path )
        
        # Setting W matrix
        set_w_matrix(model, W)


    # Last CheckPoint (resuming training)
    last_ckpt = model.find_last_saved_ckpt()
    if last_ckpt is not None:
        print(f' - Found ckpt: {last_ckpt}')
    

    # Training DataLoaders
    dl_trn = DataLoader(
            ds_trn,
            batch_size=args.BATCH_SIZE,
            num_workers=args.N_WORKERS,
            shuffle=True,
        )

    dl_val = DataLoader(
        ds_val,
        batch_size=args.BATCH_SIZE,
        num_workers=args.N_WORKERS,
        shuffle=True,
    )

    use_arcface_multigpu = (args.criterion_name.lower() == 'arcface_multigpu')
    n_gpus = len(BACKBONE_GPUS)
    # Setting up trainer
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_last=True,
        save_top_k=3,
        dirpath=None,
        filename='FacebookModel_{epoch:2d}_{trn_loss_epoch:0.04f}_{trn_acc_epoch:0.04f}_{val_loss_epoch:0.04f}_{val_acc_epoch:0.04f}'
    )

    if last_ckpt is not None:
        print(' - Restarting from:', last_ckpt)

    trainer = pl.Trainer(
        default_root_dir=model.checkpoint_base_path,
        gradient_clip_val=model.clip_grad_norm,
        gradient_clip_algorithm='norm',
        limit_val_batches=0.05,
        gpus=BACKBONE_GPUS[:1] if use_arcface_multigpu else BACKBONE_GPUS,
        check_val_every_n_epoch=1,
        accumulate_grad_batches=args.n_steps_grad_update,
        precision=args.precision,
        max_epochs=1000,
        min_epochs=500,
        reload_dataloaders_every_n_epochs=1,
        auto_lr_find=False,
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=last_ckpt,
        accelerator=(None if (use_arcface_multigpu or n_gpus<=1) else args.accelerator),
        auto_select_gpus=False,
        replace_sampler_ddp=False if (use_arcface_multigpu or n_gpus<=1) else (args.accelerator=='ddp'),
        num_sanity_val_steps=2,
    )

    # Fitting
    trainer.fit(model, dl_trn, dl_val)



if __name__ == '__main__':
    typer.run(main)
