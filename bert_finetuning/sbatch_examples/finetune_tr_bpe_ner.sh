#!/bin/sh

# this is run with "sbatch finetune_bpe_ner.sh", from the academic-budget-bert directory.

#SBATCH --job-name=finetune_bpe_ner
#SBATCH --output=finetune_bpe_ner.out
#SBATCH --error=finetune_bpe_ner.err
#SBATCH --partition=killable
#SBATCH --gpus=3
#SBATCH --mem=50000

python run_ner.py \
  --model_name_or_path ./training_bert_bpe_tr_02_out/training_bert_bpe_tr_02_dir/epoch1000000_step2279/epoch1000000_step2279 \
  --tokenizer_name bpe-tr-16k-tokenizer \
  --dataset_name polyglot_ner \
  --dataset_config_name tr \
  --text_column_name words \
  --label_column_name ner \
  --output_dir ./bpe-tr-out/ner_10000_steps \
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 8 \
  --learning_rate 5e-5 \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --lr_scheduler_type polynomial \
  --warmup_steps 50 \
  --seed 1332
