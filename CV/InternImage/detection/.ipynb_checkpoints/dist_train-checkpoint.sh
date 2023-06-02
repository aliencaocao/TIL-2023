#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=1 --master_port=63667 train.py configs/coco/cascade_internimage_l_fpn_3x_coco_custom.py --launcher pytorch --local-rank=0
