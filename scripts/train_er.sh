#!/bin/bash
#SBATCH --job-name=train_er
#SBATCH --output=output_train_er.log
#SBATCH --error=error_train_er.log
#SBATCH --nodes=1                           # This needs to match Trainer(num_nodes=...)
#SBATCH --ntasks-per-node=4               # This needs to match Trainer(devices=...)
#SBATCH --cpus-per-task=12      # 每个任务使用 12 个 CPU 核⼼  max 72
#SBATCH --mem=150G                # 每个节点使⽤128G内存       max 500
#SBATCH --partition=gpujl     # 队列名称为gpujl
#SBATCH --gres=gpu:4              # 请求 4 块 GPU（如果需要）   max 4

## **确保 Slurm 能找到 conda**
#source /home/wangyu/anaconda3/etc/profile.d/conda.sh
#
## **激活 conda 虚拟环境**
#conda activate /home/wangyu/anaconda3/envs/t2tco

# **检查 Python 环境是否正确**
which python
python --version
#nvidia-smi

# 你的任务命令
echo "Starting job on `hostname`"



python train.py \
  --task "mis" \
  --project_name "consistency_co" \
  --wandb_logger_name "mis_consistency_er_128" \
  --do_train \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "results" \
  --training_split "data/mis/er_700_800_train/*gpickle" \
  --training_split_label_dir "data/mis/er_700_800_train_labels" \
  --validation_split "data/mis/er/er_700_800_test/*gpickle" \
  --batch_size 4 \
  --num_epochs 50 \
  --hidden_dim 128 \
  --validation_examples 128 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 1 \
  --use_activation_checkpoint \
  --boundary_func truncate\
  --alpha 0.5 \
  --consistency \
  --offline