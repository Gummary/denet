#!/bin/bash

if [ $# != 2 ] ; then
  echo "USAGE: $0 NET EXPName"
  echo " e.g.: $0 RCAN exp1"
  exit 1;
fi



python main.py \
    --model $1 \
    --dataset UCLand \
    --scale 4 \
    --batch_size 16 \
    --max_steps 100000 \
    --dataset_root dataset/UCMerced_LandUse/ \
    --save_root ./output/$1/$2 \
    --ckpt_root ./pt/$1/$2