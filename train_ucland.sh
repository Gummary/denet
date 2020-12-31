#!/bin/bash

python main.py \
    --model RCAN \
    --dataset UCLand \
    --scale 4 \
    --batch_size 16 \
    --max_steps 100000 \
    --dataset_root dataset/UCMerced_LandUse/