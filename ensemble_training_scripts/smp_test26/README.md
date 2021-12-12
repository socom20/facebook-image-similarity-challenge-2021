# Test 26

- Move the files to the project's root directory.

- Start the training using the following command:
```
python3 trainer.py --n-facebook-samples 25000 --n-imagenet-samples 25000 --batch-size 200  --qry-augs-intensity 0.5
```

- After having reached a low loss value (trn_loss < ~2.0), run the next iteration of Drip Training:
```
python3 trainer.py --n-facebook-samples 50000 --n-imagenet-samples 50000 --batch-size 100  --qry-augs-intensity 0.5 --build-w-matrix ./RUN_50k_samples/lightning_logs/version_0/checkpoints/last.ckpt
```


- After having reached a low loss value (trn_loss < ~2.0), run the next iteration of Drip Training:
```
python3 trainer.py --n-facebook-samples 100000 --n-imagenet-samples 100000 --batch-size 64  --qry-augs-intensity 0.6 --build-w-matrix ./RUN_100k_samples/lightning_logs/version_0/checkpoints/last.ckpt
```

- After having reached a low loss value (trn_loss < ~2.0), run the next iteration of Drip Training:
```
python3 trainer.py --n-facebook-samples 250000 --n-imagenet-samples 250000 --batch-size 64  --qry-augs-intensity 0.7 --build-w-matrix ./RUN_200k_samples/lightning_logs/version_0/checkpoints/last.ckpt
```

- After having reached a low loss value (trn_loss < ~1.0), run the next iteration of Drip Training:
```
python3 trainer.py --n-facebook-samples 500000 --n-imagenet-samples 500000 --batch-size 64  --qry-augs-intensity 0.75 --build-w-matrix ./RUN_500k_samples/lightning_logs/version_0/checkpoints/last.ckpt
```

- After having reached a low loss value (trn_loss < ~1.0), run the next iteration of Drip Training:
```
python3 trainer.py --n-facebook-samples 1000000 --n-imagenet-samples 500000 --batch-size 64  --qry-augs-intensity 0.75 --build-w-matrix ./RUN_1000k_samples/lightning_logs/version_0/checkpoints/last.ckpt
```

- Output dir: `./RUN_1500k_samples`

- Validate the training using the following notebook:
```
inference_test26.ipynb
```
