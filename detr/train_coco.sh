#!/bin/bash

export OMP_NUM_THREADS=1 &&
python DETR/main.py \
    --batch_size=1 \
    --coco_path /data/coco_detr

