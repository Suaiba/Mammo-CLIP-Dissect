#!/bin/bash

python ./src/single_gpu_train_classifier.py \
  --data-dir '/storage' \
  --img-dir 'VinDR_MammoCLIP/images_png' \
  --csv-file 'VinDR-data/new_vindr_detection_v1_folds.csv' \
  --clip_chk_pt_path "/storage/BreastCLIP_models/b5-model-best-epoch-7.tar" \
  --data_frac 1.0 \
  --dataset 'VinDr' \
  --arch 'upmc_breast_clip_det_b5_period_n_ft' \
  --label "density" \
  --epochs 30 \
  --batch-size 8 \
  --num-workers 0 \
  --print-freq 10000 \
  --log-freq 500 \
  --running-interactive 'n' \
  --n_folds 1 \
  --lr 5.0e-5 \
  --weighted-BCE 'y' \
  --balanced-dataloader 'n' \
  --inference-mode 'load' \
  --checkpoints '/storage2/MammoCLIP_classfier_checkpoints'