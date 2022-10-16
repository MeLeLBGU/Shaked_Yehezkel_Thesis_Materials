#!/bin/sh

# this is run with "sbatch finetune_sage_mrpc.sh", from the academic-budget-bert directory.

#SBATCH --job-name=finetune_sage_mrpc
#SBATCH --output=finetune_sage_mrpc.out
#SBATCH --error=finetune_sage_mrpc.err
#SBATCH --partition=killable
#SBATCH --nodelist=rack-omerl-g01
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=150Mib

python run_glue.py \
  --model_name_or_path sage-16-c500k-tokenizer-out/sage-16-c500k-tokenizer-/epoch1000000_step5640/ \
  --tokenizer_name sage-16-c500k \
  --task_name MRPC \
  --max_seq_length 128 \
  --output_dir ./sage-16k-out/mrpc \
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  --do_predict \
  --evaluation_strategy steps \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 16 \
  --learning_rate 5e-5 \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --lr_scheduler_type polynomial \
  --warmup_steps 50 \
  --seed 1332
