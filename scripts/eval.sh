#!/bin/bash
# Set the fold number
FOLD=0
# Set the checkpoint file name
CKPT="b5-model-best-epoch-7.tar"
# Set the directory path
DIR="/cephfs"
# Construct the full checkpoint path
FULL_CKPT="$DIR/$CKPT"
# Execute the Python script with the specified arguments
python ./src/zero_shot_eval_clip.py \
  --config-name zs_clip.yaml hydra.run.dir=$DIR model.clip_check_point=$FULL_CKPT