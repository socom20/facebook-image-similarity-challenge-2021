
# Downloading Datasets:

- Download the datasets and place them inside ```all_datasets/dataset```.
- Follow the guide ```all_datasets/dataset/readme.txt``` to download the datasets and rename the samples (ImageNet and Google's deep fake datasets).
- Image ```all_datasets/training_datasets_structure.png``` shows the dataset structure needed to do training (```all_datasets/dataset_jpg_384x384``` will be created once the first training call starts).


# Model Training: Drip Training procedure


In order to do a fast training you must follow:

1. Move all the scripts to the project-root directory.
- File ```model_config.py``` contains all the model's configuration hyperparameters, some hyperparameters are rewritten by ```trainer.py``` in order to improve accessibility.

2. To start a fast training, you can do a first training using only 50k samples. In this case the training starts using only 50k images from Facebook's training dataset.
- This first iteration will be completed in less than an hour since you are using only 50k classes in the ArcFace head.
- The checkpoints will be saved in ```./TEST26_50k_nobias_l1/lightning_logs/version_0/checkpoints```.
- You can also explore the Tensorboard logs of the training by doing: ```tensorboard logdir=./TEST26_50k_nobias_l1/lightning_logs```
- run: 
```
python3 trainer.py --n-facebook-samples 50000 --n-imagenet-samples 0 --n-frm-face-samples 0 --batch-size 200 --qry-augs-intensity 0.5
```

3. Once the validation loss drops below val_loss<1.0 (or trn_loss<2.0), you can increase the number of classes in the arcface head.
- In order to continue the training using a more classes, you need to build the centroids matrix for the new target. 
- To create the centriods matrix (W matrix), you need to provide the last checkpoint obtained from the previous iteration using the argument ```--build-w-matrix CKPT_PATH```.
- You can include 50k new images from the ImageNet dataset by doing:
```
python3 trainer.py --n-facebook-samples   50000 --n-imagenet-samples  50000 --n-frm-face-samples 0 --batch-size 200 --qry-augs-intensity 0.5 --build-w-matrix ./TEST26_50k_nobias_l1/lightning_logs/version_0/checkpoints/last.ckpt
```
- The new checkpoints will be saved in ```./TEST26_100k_nobias_l1/lightning_logs/version_0/checkpoints```.
- The training will start from epoch 0.
- This time, the training loss will start from a small value trn_loss ~= 2.0.

4. Once the validation loss goes below val_loss<1.0 (or trn_loss<2.0), you can increase the number of classes in the arcface head again:
```
python3 trainer.py --n-facebook-samples  100000 --n-imagenet-samples 100000 --n-frm-face-samples 0 --batch-size 100 --qry-augs-intensity 0.50 --build-w-matrix ./TEST26_100k_nobias_l1/lightning_logs/version_0/checkpoints/last.ckpt
```

5. You can continue this procedure increasing the number of classes in the ArcFace head and also increacing the augmentations intensity. You may also want to change the batch size.
- Below is an example to complete the training procedure :
```
python3 trainer.py --n-facebook-samples  250000 --n-imagenet-samples 250000 --n-frm-face-samples 0 --batch-size 84  --qry-augs-intensity 0.60 --build-w-matrix ./TEST26_200k_nobias_l1/lightning_logs/version_0/checkpoints/last.ckpt
python3 trainer.py --n-facebook-samples  750000 --n-imagenet-samples 250000 --n-frm-face-samples 0 --batch-size 84  --qry-augs-intensity 0.65 --build-w-matrix ./TEST26_500k_nobias_l1/lightning_logs/version_0/checkpoints/last.ckpt
python3 trainer.py --n-facebook-samples 1000000 --n-imagenet-samples 500000 --n-frm-face-samples 0 --batch-size 64  --qry-augs-intensity 0.75 --build-w-matrix ./TEST26_1000k_nobias_l1/lightning_logs/version_0/checkpoints/last.ckpt
```


# Model Inference:

- To run inference you must run ```inference.py``` script.
- You need to provide the chekpoint path, for example: ```./TEST26_100k_nobias_l1/lightning_logs/version_0/checkpoints/last.ckpt```:

```
python3 inference.py --batch-size 200 --checkpoint-path ./TEST26_100k_nobias_l1/lightning_logs/version_0/checkpoints/last.ckpt
```

- When the inference finishes, the phase 1 submission file will be saved in the same location of the ckpt file, for example: ```./TEST26_100k_nobias_l1/lightning_logs/version_0/checkpoints/last_160x160_REF.h5```
- The mAP score will be calculated automatically using the ground truth samples provided with the dataset.






