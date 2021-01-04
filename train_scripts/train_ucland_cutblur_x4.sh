#!/bin/bash

if [ $# != 2 ]; then
  echo "USAGE: $0 NET EXPName"
  echo " e.g.: $0 RCAN exp1"
  exit 1
fi

# Iterator 80K times, decay lr at 20K, 40K, 60K

python main.py \
--model $1 \
--dataset UCLand \
--scale 4 \
--batch_size 2 \
--max_steps 640000 \
--decay 200-400-600 \
--dataset_root dataset/UCMerced_LandUse/ \
--save_root ./output/$1/$2 \
--ckpt_root ./pt/$1/$2 \
--augs cutblur \
--alpha 0.7 \
--prob 0.5
