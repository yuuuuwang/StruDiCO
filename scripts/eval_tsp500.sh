#!/bin/bash

which python
python --version
# nvidia-smi
export CUDA_VISIBLE_DEVICES=1,2,0,3

python train.py \
  --task "tsp" \
  --project_name "consistency_co_test_500" \
  --wandb_logger_name "tsp_500" \
  --do_test \
  --storage_path "./" \
  --test_split "data/tsp_test/tsp500_lkh_50000_16.54811.txt" \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 5 \
  --two_opt_iterations 5000 \
  --ckpt_path "ckpts/tsp500.ckpt" \
  --consistency \
  --sparse_factor 50 \
  --resume_weight_only \
  --parallel_sampling 1 \
  --sequential_sampling 1 \
  --rewrite \
  --rewrite_steps 5 \
  --rewrite_ratio 0.3 \
  --m2 0.7 \
  --m1 0.7 \
  --offline

# === Batch experiments === "results/tsp_consistency_500_alpha_0.5_bz6_pretrain/gqag42ot/checkpoints/epoch=45-step=245318.ckpt" \
#run_experiment tsp500 1 5 0 0.7 0
#run_experiment tsp500 1 5 0 0.7 0.3
#run_experiment tsp500 1 5 5000 0.7 0
#run_experiment or_tsp500 1 5 5000 0.7 0.3 "results/tsp_consistency_500_alpha_0.5_bz6_pretrain/gqag42ot/checkpoints/epoch=45-step=245318.ckpt"

#run_experiment tsp500 4 5 0 0.5 0
#run_experiment tsp500 4 5 0 0.5 0.3
#run_experiment tsp500 4 5 5000 0.8 0
#run_experiment tsp500 4 5 5000 0.8 0.3
