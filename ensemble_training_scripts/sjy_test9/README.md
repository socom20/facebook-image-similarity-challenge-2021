# Test 9

- Move the files to the project's root directory.

- Start the training using the following command:
```
python3 trainer.py --n-facebook-samples 40000 --n-imagenet-samples 0 --batch-size 200  --qry-augs-intensity 0.5
```

- After having reached a low loss value (trn_loss < ~2.0), run the next iteration of Drip Training:
```
python3 trainer.py --n-facebook-samples 80000 --n-imagenet-samples 0 --batch-size 100  --qry-augs-intensity 0.5 --build-w-matrix ./RUN_40k_samples/lightning_logs/version_0/checkpoints/last.ckpt
```


- After having reached a low loss value (trn_loss < ~2.0), run the next iteration of Drip Training:
```
python3 trainer.py --n-facebook-samples 150000 --n-imagenet-samples 0 --batch-size 100  --qry-augs-intensity 0.6 --build-w-matrix ./RUN_80k_samples/lightning_logs/version_0/checkpoints/last.ckpt
```

- After having reached a low loss value (trn_loss < ~2.0), run the next iteration of Drip Training:
```
python3 trainer.py --n-facebook-samples 300000 --n-imagenet-samples 0 --batch-size 64  --qry-augs-intensity 0.7 --build-w-matrix ./RUN_150k_samples/lightning_logs/version_0/checkpoints/last.ckpt
```

- After having reached a low loss value (trn_loss < ~1.0), run the next iteration of Drip Training:
```
python3 trainer.py --n-facebook-samples 500000 --n-imagenet-samples 100000 --batch-size 64  --qry-augs-intensity 0.75 --build-w-matrix ./RUN_300k_samples/lightning_logs/version_0/checkpoints/last.ckpt
```

- After having reached a low loss value (trn_loss < ~1.0), run the next iteration of Drip Training:
```
python3 trainer.py --n-facebook-samples 1000000 --n-imagenet-samples 800000 --batch-size 64  --qry-augs-intensity 0.75 --build-w-matrix ./RUN_600k_samples/lightning_logs/version_0/checkpoints/last.ckpt
```

- Output dir: `./RUN_1800k_samples`

- Validate the training using the following notebook:
```
inference_test9.ipynb
```
