export CUDA_VISIBLE_DEVICES=0,2,3,1


which python
python --version
# nvidia-smi

echo "Starting job on $(hostname)"

python train.py \
 --task "mis" \
--project_name "consistency_co_test" \
--wandb_logger_name "mis_rb" \
--do_test \
--storage_path "./" \
--test_split "data/mis/rb/rb200_300_test/*gpickle" \
--test_split_label_dir "data/mis/rb/rb200_300_test_label" \
--inference_schedule "cosine" \
--inference_diffusion_steps 5 \
--ckpt_path "ckpts/mis_rb.ckpt" \
--consistency \
--resume_weight_only \
--parallel_sampling 1 \
--sequential_sampling 1 \
--rewrite \
--rewrite_steps 5 \
--rewrite_ratio 0.3 \
--m2 0.3 \
--m1 0.3 \
--offline

