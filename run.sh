#!/bin/bash

# python main.py --path_opt cfgs/normal_coco_res101.yml --batch_size 100 --cuda True --num_workers 10 --max_epoch 50 --mGPUs True
# python main.py --path_opt cfgs/normal_coco_res101.yml --batch_size 1 --cuda True --num_workers 1 --max_epoch 50 --mGPUs True

CUDA_VISIBLE_DEVICES=0 python main.py --path_opt cfgs/robust_coco.yml --batch_size 100 --cuda True --num_workers 10 --max_epoch 50 --relation_type spatial123 --graph_attention True