#!/bin/bash

if [ $# < 2 ]; then
  echo "USAGE: $0 NET EXPName [ARGS]"
  echo " e.g.: $0 RCAN exp1"
  exit 1
fi

# shellcheck disable=SC1083
NET=$1_DC

# Iterator 80K times, decay lr at 20K, 40K, 60K

python main.py \
--model $NET \
--dataset UCLand \
--scale 4 \
--batch_size 2 \
--max_steps 640000 \
--decay 200-400-600 \
--dataset_root dataset/UCMerced_LandUse/ \
--save_root ./output/$NET/$2 \
--ckpt_root ./pt/$NET/$2 \
--scale 4 \
"${@:3}"
