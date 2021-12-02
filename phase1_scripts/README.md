# PHASE 1: Inference

In order to reproduce Phase 1 submissions, you have to follow these steps:

1. Move ```./phase1_scripts/*``` to ```./``` (being ```./``` the project's root directory)

2. Move/Copy/Symlink the competition's raw images to: ```./all_datasets/dataset```. <br />
   Read: <br />
   ```./all_datasets/dataset/readme.md``` for details about the dataset structure.
   ```./all_datasets/inference_datasets_structure.png``` show de entire dataset structure.
	
   If you are planning to run phase1 and phase2 inferences, "./all_datasets/dataset/query_images" should
   contain images from "Q00000.jpg" to "Q99999.jpg".
   
3. Execute sequentially all inference notebooks:
```
    ./inference_test5_epoch51.ipynb
    ./inference_test9_epoch9.ipynb
    ./inference_test10_epoch9.ipynb
    ./inference_test10_epoch37.ipynb
    ./inference_test19_epoch67.ipynb
    ./inference_test22_epoch15.ipynb
    ./inference_test24_epoch79.ipynb
    ./inference_test25_1M_epoch14.ipynb
    ./inference_test25_1M_epoch16.ipynb
    ./inference_test26_500k_epoch27.ipynb
    ./inference_test26_1500k_epoch13.ipynb
    ./inference_test26_1500k_epoch17.ipynb
```

When running these scripts, datasets ```./all_datasets/dataset_jpg_384x384``` and ```./all_datasets/dataset_jpg_512x512``` will be automatically created.


4. Finally, run ```submissions_v8.ipynb```.

5. The submission files will be saved inside "./submissions" (*.h5 and *.csv files):


### - Description track outputs:

```
./submissions/submission_V8.h5
./submissions/submission_V8_GT.h5      # Same as "./submissions/submission_V8.h5" but including the Grand Truth samples.
./submissions/submission_V8_NORM.h5    # A normalized version of "./submissions/submission_V8.h5"
./submissions/submission_V8_NORM_GT.h5 # Same as "./submissions/submission_V8_NORM.h5" but including the Grand Truth samples.
```


### -  Matching track outputs:
```
./submissions/match_submission_V8.csv
./submissions/match_submission_V8_GT.csv
```

