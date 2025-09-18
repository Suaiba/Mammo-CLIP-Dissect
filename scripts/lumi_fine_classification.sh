#!/bin/bash

python ./src/train_classifier.py \
  --data-dir '/scratch/project_465001915/salahudd' \
  --img-dir 'VinDR_mammoclip/images_png' \
  --csv-file 'MammoCLIP_csvs/new_vindr_detection_v1_folds.csv' \
  --clip_chk_pt_path "/scratch/project_465001915/salahudd/b5-model-best-epoch-7.tar" \
  --data_frac 1.0 \
  --dataset 'VinDr' \
  --arch 'upmc_breast_clip_det_b5_period_n_ft' \
  --label "cancer_birads" \
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
  --inference-mode 'n'