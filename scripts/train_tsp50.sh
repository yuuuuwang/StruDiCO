#!/bin/bash
#SBATCH --job-name=t50abs      # 任务名称
#SBATCH --output=output_train_tsp50_abs.log       # 标准输出日志文件
#SBATCH --error=error_train_tsp50_abs.log         # 错误日志文件
#SBATCH --ntasks=1                # 总共 1 个任务
#SBATCH --cpus-per-task=4       # 每个任务使用 4 个 CPU 核⼼
#SBATCH --mem=64G                # 每个任务使⽤4G内存
#SBATCH --partition=gpujl     # 队列名称为gpujl
#SBATCH --gres=gpu:4              # 请求 1 块 GPU（如果需要）

## **确保 Slurm 能找到 conda**
#source /home/wangyu/anaconda3/etc/profile.d/conda.sh
#
## **激活 conda 虚拟环境**
#conda activate /home/wangyu/anaconda3/envs/ml4tsp

# **检查 Python 环境是否正确**
which python
python --version

# 你的任务命令
echo "Starting job on `hostname`"

export CUDA_VISIBLE_DEVICES=2,3,1
#
python train.py \
  --task "tsp" \
  --project_name "consistency_co" \
  --wandb_logger_name "tsp_consistency_50_alpha_0.5_not_same_trial_pretrain" \
  --do_train \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "results/" \
  --training_split "data/tsp_train/tsp50_uniform_1.28m.txt" \
  --validation_split "data/tsp/tsp50_concorde.txt" \
  --test_split "data/tsp_test/tsp50_lkh_500_5.68759.txt" \
  --batch_size 32 \
  --num_epochs 50 \
  --hidden_dim 256 \
  --validation_examples 1280 \
  --diffusion_schedule "linear" \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 1 \
  --two_opt_iterations 0 \
  --alpha 0.5 \
  --boundary_func truncate \
  --consistency \
  --rewrite \
  --offline \
  --ckpt_path "ckpts/tsp50.ckpt" \
  --resume_weight_only