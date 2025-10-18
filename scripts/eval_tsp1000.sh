#!/bin/bash

which python
python --version
# nvidia-smi
export CUDA_VISIBLE_DEVICES=1,2,0,3

python train.py \
  --task "tsp" \
  --project_name "consistency_co_test_1000" \
  --wandb_logger_name "tsp_1000" \
  --do_test \
  --storage_path "./" \
  --test_split "data/tsp_test/tsp1000_concorde_23.11812.txt" \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 5 \
  --two_opt_iterations 5000 \
  --ckpt_path "ckpts/tsp1000.ckpt" \
  --consistency \
  --sparse_factor 100 \
  --resume_weight_only \
  --parallel_sampling 1 \
  --sequential_sampling 1 \
  --rewrite \
  --rewrite_steps 5 \
  --rewrite_ratio 0.2 \
  --m2 0.8 \
  --m1 0.8 \
  --offline
