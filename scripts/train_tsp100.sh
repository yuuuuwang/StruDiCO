#!/bin/bash
#SBATCH --job-name=t100_bt2
#SBATCH --output=output_train_tsp100_bt2.log
#SBATCH --error=error_train_tsp100_bt2.log
#SBATCH --nodes=1                           # This needs to match Trainer(num_nodes=...)
#SBATCH --ntasks-per-node=4               # This needs to match Trainer(devices=...)
#SBATCH --cpus-per-task=12      # 每个任务使用 12 个 CPU 核⼼  max 72
#SBATCH --mem=300G                # 每个节点使⽤128G内存       max 500
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


srun python train.py \
  --task "tsp" \
  --project_name "consistency_co" \
  --wandb_logger_name "tsp_consistency_100_alpha_0.5_pretrain" \
  --do_train \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "results/" \
  --training_split "data/tsp_train/tsp100_train_concorde.txt" \
  --validation_split "data/tsp/tsp100_concorde.txt" \
  --test_split "data/tsp_test/tsp100_concorde_7.75585.txt" \
  --batch_size 12 \
  --num_epochs 50 \
  --validation_examples 1280 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 1 \
  --two_opt_iterations 0 \
  --alpha 0.5 \
  --boundary_func truncate \
  --consistency \
  --offline  \
  --ckpt_path "results/tsp_consistency_100_alpha_0.5_pretrain/hslylpw3/checkpoints/last.ckpt"
