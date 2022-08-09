#!/bin/bash
CONFIG=$1
GPUS=$2
RESUME=${3:-0}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}
torchrun --nnodes=$NNODES \
	--node_rank=$NODE_RANK \
	--nproc_per_node=$GPUS \
	--master_addr=$MASTER_ADDR \
	--master_port=$MASTER_PORT \
	tools/main.py --config $CONFIG --multigpu 1 --resume $RESUME
