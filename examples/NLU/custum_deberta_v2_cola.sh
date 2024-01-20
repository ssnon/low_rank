export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./output"
CUDA_VISIBLE_DEVICES=0,1\ python examples/text-classification/run_glue_head.py \
--model_name_or_path microsoft/deberta-v2-xxlarge \
--task_name cola \
--do_train \
--do_eval \
--max_seq_length 32 \
--per_device_train_batch_size 1 \
--learning_rate 1.3e-4 \
--num_train_epochs 16 \
--output_dir $output_dir/headwiseB/cola \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/headwiseB/cola/log \
--fp16 \
--evaluation_strategy steps \
--eval_steps 100 \
--save_strategy steps \
--save_steps 100 \
--save_total_limit 5 \
--warmup_steps 1000 \
--cls_dropout 0.1 \
--apply_lora \
--lora_r 1 \
--lora_alpha 32 \
--seed 0 \
--weight_decay 0.01 \
--ex_type lora_headwiseB
--auto_find_batch_size True