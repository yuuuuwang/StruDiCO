#!/bin/bash

which python
python --version
# nvidia-smi

export CUDA_VISIBLE_DEVICES=1,2,0,3

echo "Starting job on $(hostname)"

python train.py \
--task "tsp" \
--project_name "consistency_co_test_100" \
--wandb_logger_name "tsp_100" \
--do_test \
--storage_path "./" \
--test_split "data/tsp_test/tsp100_concorde_7.75585.txt" \
--inference_schedule "cosine" \
--inference_diffusion_steps 3 \
--two_opt_iterations 5000 \
--ckpt_path "ckpts/tsp100.ckpt" \
--consistency \
--resume_weight_only \
--parallel_sampling 1 \
--sequential_sampling 1 \
--rewrite \
--rewrite_steps 3 \
--rewrite_ratio 0.4 \
--m2 0.8 \
--m1 0.8 \
--offline

# === Batch experiments === "results/tsp_consistency_100_alpha_0.5_pretrain/hslylpw3/checkpoints/epoch=47-step=1501968.ckpt" \
#run_experiment tsp100 1 3 0 0.5 0 "results/tsp_consistency_100_alpha_0.5_pretrain/hslylpw3/checkpoints/epoch=47-step=1501968.ckpt"
#run_experiment tsp100 1 3 0 0.5 0.4 "results/tsp_consistency_100_alpha_0.5_pretrain/hslylpw3/checkpoints/epoch=47-step=1501968.ckpt"
#run_experiment tsp100 1 3 5000 0.8 0 "results/tsp_consistency_100_alpha_0.5_pretrain/hslylpw3/checkpoints/epoch=47-step=1501968.ckpt"
#run_experiment or_tsp100 1 3 5000 0.8 0.4 "results/tsp_consistency_100_alpha_0.5_pretrain/hslylpw3/checkpoints/epoch=47-step=1501968.ckpt"
