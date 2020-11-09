#!/bin/bash

python main.py --path_opt cfgs/normal_coco_res101.yml --batch_size 100 --cuda True --num_workers 10 --max_epoch 50 --mGPUs True
# python main.py --path_opt cfgs/normal_coco_res101.yml --batch_size 2 --cuda True --num_workers 1 --max_epoch 50 --mGPUs True