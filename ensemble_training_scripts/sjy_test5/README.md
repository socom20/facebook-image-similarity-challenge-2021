# Test 5

- Move the files to the project's root directory.

- You can modify in scripts: `BATCH_SIZE`, `N_WORKERS` and `N_GPUS`

- Start the training using the following command (image resolution 224x224):

```
python3 Facebook_model_v15_augsv5_224x224.py
```

- Wait until the loss goes down (trn_loss < ~2.0), and run the following command (image resolution 512x512):

```
python3 Facebook_model_v15_augsv5_512x512.py
```

- Output dir: `./TEST15_arcface`

- Validate the training using the following notebook:
```
inference_test5.ipynb
```
