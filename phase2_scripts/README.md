# PHASE 2: Inference

In order to reproduce Phase 2 submissions, you have to follow these steps:

1. Move ```./pahse2_scripts/*``` to ```./``` (being ```./``` the project's root directory)

2. Move/Copy/Symlink the competition's raw images to: "./all_datasets/dataset". <br />
   Read: <br />
    ```./all_datasets/dataset/readme.txt" for details about the dataset structure.```
    ```./all_datasets/inference_datasets_structure.png" show de entire dataset structure.```
    
   If you are planning to run phase1 and phase2 inferences, ```./all_datasets/dataset/query_images``` should
   contain images from ```Q00000.jpg``` to ```Q99999.jpg```.
   
3. Execute tha bash script: "./run_all_inferences.sh". This bash script will run the same models from Phase 1.

4. Finally, run ```p2_submissions_v9.ipynb```.

5. The submission files will be saved inside "./submissions" (*.h5 and *.csv files):

### - Description track outputs:
```
./submissions/submission_V9_P2.h5
./submissions/submission_V9_NORM_P2.h5"  # (best submission) A normalized version of "./submissions/submission_V9_P2.h5
```

### - Matching track outputs:
```
./submissions/match_submission_V9_P2.csv
```


