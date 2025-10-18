export CUDA_VISIBLE_DEVICES=0,1,2,3


python train.py \
   --task "tsp" \
   --project_name "consistency_co_test" \
   --wandb_logger_name "tsp_100000" \
   --do_test \
   --storage_path "./" \
   --test_split "tsp100000_test.txt" \
   --inference_schedule "cosine" \
   --inference_diffusion_steps 5 \
   --two_opt_iterations 5000 \
   --ckpt_path 'ckpts/tsp10000.ckpt' \
   --fp16 \
   --consistency \
   --sparse_factor 100 \
   --resume_weight_only \
   --parallel_sampling 1 \
   --sequential_sampling 1 \
   --guided \
   --rewrite_steps 5 \
   --rewrite_ratio 0.2 \
   --offline
