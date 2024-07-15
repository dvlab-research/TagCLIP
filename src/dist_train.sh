#!/usr/bin/env bash

CONFIG=$1
WORK_DIR=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 -m torch.distributed.launch  --nproc_per_node=8 --master_port=$((RANDOM + 10000)) \
    src/train.py $CONFIG --work-dir=$WORK_DIR --launcher pytorch ${@:3} 