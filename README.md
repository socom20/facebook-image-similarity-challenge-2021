# Facebook AI Image Similarity Challenge: Descriptor Track


This repository contains the code for our solution to the Facebook AI Image Similarity Challenge: Descriptor Track, hosted on DrivenData: <br />
https://www.drivendata.org/competitions/85/competition-image-similarity-2-final/leaderboard/

The solution is described in the paper hosted on arXiv.org: <br /> 
https://arxiv.org/abs/2112.03415v1


## Environment Setup
To run the models, you must first configure the environment:

- Download the datasets following the [README.md](./all_datasets) instructions at: ```all_datasets/README.md```

- Copy some query images. Go to the directory ```FB_page_qry/``` and run: ```python3 copy_fb_query_images.py```

## Checkpoints

Download all the checkpoints from my [gdrive](https://drive.google.com/drive/folders/1MnTm7OIPYuMMuc_uij7_bvT7_8NCxP-o) and locate them inside ```checkpoints/``` directory.


## Phase 1 inference

Follow the [README.md](./phase1_scripts) instructions at: ```phase1_scripts/README.md```


## Phase 2 inference

Follow the [README.md](./phase2_scripts) instructions at: ```phase2_scripts/README.md```


## Training the Final Ensemble Models 

Follow the [README.md](./ensemble_training_scripts) instructions at: ```ensemble_training_scripts/README.md```


## Training your model

Follow the [README.md](./training_scripts) instructions at: ```training_scripts/README.md```
