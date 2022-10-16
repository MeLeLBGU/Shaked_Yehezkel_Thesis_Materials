#!/bin/sh

# this is run with "sbatch finetune_tr_bpe_classification.sh", from the academic-budget-bert directory.

#SBATCH --job-name=finetune_tr_part_bpe_classification
#SBATCH --output=finetune_tr_part_bpe_classification.out
#SBATCH --error=finetune_tr_part_bpe_classification.err
#SBATCH --partition=killable
#SBATCH --nodelist=rack-omerl-g01
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=150Mib

python run_turkish_classification.py \
  --model_name_or_path ./training_bert_bpe_tr_02_out/training_bert_bpe_tr_02_dir/epoch1000000_step2279/epoch1000000_step2279 \
  --tokenizer_name bpe-tr-16k-tokenizer  \
  --output_dir ./tr_classifications/bpe_out \
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  --do_predict \
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
