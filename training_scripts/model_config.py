class Args():
    # Senttng default parameters
    
    def get_argslist(self):
        arg_list = [p for p in  dir(self) if (not p.startswith('_')) and (not p.startswith('get'))]
        return arg_list
    
    def get_argsdict(self):
        arg_list = self.get_argslist()
        arg_dict = {k : getattr(self, k) for k in arg_list}
        return arg_dict
    
    def __repr__(self):
        s = 10*' =' + ' ArgList' + 10*' =' + '\n'
        for k, v in self.get_argsdict().items():
            s += f'{k:30s}: {v}\n'
            
        return s

    def __str__(self):
        return self.__repr__()
    
    
class ArgsT26(Args):
    OUTPUT_WH            = (160, 160) # Model WH
    DATASET_WH           = (384, 384) # Dataset images (these images will be first augmentean and then resized to OUTPUT_WH)
    BATCH_SIZE           = 100
    N_WORKERS            = 20  # DataLoader workers

    BACKBONE_GPUS        = [0]
    DS_INPUT_DIR         = f'./all_datasets/dataset'  # Path where "query_images", "reference_images" and "training_images" are located

    DS_DIR               = f'{DS_INPUT_DIR}_jpg_{DATASET_WH[0]}x{DATASET_WH[1]}' # Path where the rescaled images will be saved
    ALL_FOLDERS          = ['training_images', 'imagenet_images', 'face_frames', 'query_images', 'reference_images']
    
    
    QRY_AUGS_INTENSITY   = 0.75
    
    ImgNet_SAMPLES       = './data/ImageNet_samples_v.pickle'
    FrmFaces_SAMPLES     = './data/FrameFaces_samples_v.pickle'
    
    img_norm_type        = ['simple', 'imagenet', 'centered'][1]
    model_resolution     = OUTPUT_WH[::-1]
    embedding_size       = 128 * 2
    backbone_name        = ['efficientnetv2_rw_s', 'efficientnetv2_rw_m', 'tf_efficientnetv2_l_in21ft1k',
                            'efficientnet_b4', 'tf_efficientnet_b5',
                            'nfnet_l0', 'eca_nfnet_l1', 'eca_nfnet_l2', 'eca_nfnet_l3'][-3]
    
    pretrained_bb        = True
    neck_type            = 'F'
    eps                  = 1e-6
    init_lr              = 1e-3
    use_scheduler        = True
    scheduler_name       = ['ReduceLROnPlateau', 'LinearWarmupCosineAnnealingLR'][0]
    
    sched_factor         = 0.90
    sched_patience       = 2
    
    n_warmup_epochs      = 4
    n_decay_epochs       = 100                           
    lr_prop              = 1e-3
    
    
    optimizer_name       = ['adam', 'sgd'][0]
    clip_grad_norm       = 1.0
    weight_decay         = [0.0, 1e-7][0]
    n_steps_grad_update  = 1
    criterion_name       = ['arcface', 'arcface_multigpu'][0]
    
    start_ckpt           = None
    build_w_matrix       = False
    
    do_ref_trn           = False
    ref_trn_n_epochs     = 5
    
    n_facebook_samples   = 1_000_000  # TotalSamples = 1_000_000 Facebook's training dataset
    n_imagenet_samples   =         0  # TotalSamples = 1_431_167 ImageNet
    n_frm_face_samples   =         0  # TotalSamples = 476_584 FrameFaces (Google's DeepFake dataset)
    
    arc_classnum         = n_facebook_samples + n_imagenet_samples + n_frm_face_samples
    arc_devices_v        = [0]
    arc_w_prop_v         = None
    arc_optimizer        = None
    
    arc_m                = 0.4
    arc_s                = 40.0
    arc_bottleneck       = None
    arc_weight_decay     = 0.0
    use_GeM              = True
    GeM_p                = 3.0
    GeM_opt_p            = True
    checkpoint_base_path = f'./TEST26_{int(arc_classnum/1000)}k_{backbone_name}'
    model_name           = 'arc_model'
    
    precision            = [32, 16][1]
    seed                 = None
    
    accelerator          = ['dp', 'ddp'][1]
