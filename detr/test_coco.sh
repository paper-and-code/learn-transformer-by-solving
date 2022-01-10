#!/bin/bash 
## DETR Official Code
# python DETR/main.py \
#     --batch_size 1 \
#     --no_aux_loss \
#     --eval \
#     --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
#     --coco_path /data/coco_detr

## DETR ShortVersion
python test.py \
    --batch_size 1 \
    --no_aux_loss \
    --eval \
    --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
    --coco_path /data/coco_detr