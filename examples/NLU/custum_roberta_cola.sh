export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./output"
python examples/text-classification/run_glue_head.py \
--model_name_or_path roberta-base \
--task_name cola \
--do_train \
--do_eval \
--max_seq_length 512 \
--per_device_train_batch_size 8 \
--learning_rate 4e-4 \
--num_train_epochs 10 \
--output_dir $output_dir/headwiseB_cola_roberta \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/headwiseB_cola_roberta/log \
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
--lora_alpha 16 \
--seed 0 \
--weight_decay 0.01 \
--ex_type lora_headwiseB