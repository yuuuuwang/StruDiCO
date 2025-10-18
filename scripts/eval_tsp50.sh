#!/bin/bash

which python
python --version
# nvidia-smi
export CUDA_VISIBLE_DEVICES=1,2,0,3

python train.py \
--task "tsp" \
--project_name "consistency_co_test_50" \
--wandb_logger_name "tsp_50" \
--do_test \
--storage_path "./" \
--test_split "data/tsp_test/tsp50_lkh_500_5.68759.txt" \
--inference_schedule "cosine" \
--inference_diffusion_steps 3 \
--two_opt_iterations 5000 \
--ckpt_path "ckpts/tsp50.ckpt" \
--consistency \
--resume_weight_only \
--parallel_sampling 1 \
--sequential_sampling 1 \
--rewrite \
--rewrite_steps 3 \
--rewrite_ratio 0.4 \
--m2 0.9 \
--m1 0.9 \
--offline

