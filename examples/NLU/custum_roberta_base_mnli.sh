export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./output"
python examples/text-classification/run_glue_head.py \
--model_name_or_path roberta-base \
--task_name mnli \
--do_train \
--do_eval \
--max_seq_length 512 \
--per_device_train_batch_size 8 \
--learning_rate 5e-4 \
--num_train_epochs 15 \
--output_dir $output_dir/headwiseB_mnli_roberta \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/headwiseB_mnli_roberta/log \
--evaluation_strategy epoch \
--save_strategy epoch \
--warmup_ratio 0.06 \
--apply_lora \
--lora_r 1 \
--lora_alpha 16 \
--seed 0 \
--weight_decay 0.01 \
--ex_type lora_headwiseB