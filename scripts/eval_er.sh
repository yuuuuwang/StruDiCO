export CUDA_VISIBLE_DEVICES=1,3,0,2

which python
python --version
# nvidia-smi

python train.py \
 --task "mis" \
--project_name "consistency_co_test" \
--wandb_logger_name "mis_er" \
--do_test \
--storage_path "./" \
--test_split "data/mis/er/er_700_800_test/*gpickle" \
--inference_schedule "cosine" \
--inference_diffusion_steps 5 \
--ckpt_path "ckpts/mis_er.ckpt" \
--consistency \
--resume_weight_only \
--parallel_sampling 1 \
--sequential_sampling 1 \
--hidden_dim 128 \
--rewrite \
--rewrite_steps 5 \
--rewrite_ratio 0.3 \
--m2 0.3 \
--m1 0.3 \
--offline

