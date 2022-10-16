#!/bin/sh

# this is run with "sbatch pretrain_bert_from_checkpoint.sh", from the academic-budget-bert directory.

#SBATCH --job-name=pretrain_bert_bpe_16k_from_checkpoint
#SBATCH --output=pretrain_bert_bpe_16k_from_checkpoint.out
#SBATCH --error=pretrain_bert_bpe_16k_from_checkpoint.err
#SBATCH --partition=killable
#SBATCH --nodelist=rack-omerl-g01
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=150Mib

deepspeed run_pretraining.py \
  --layer_norm_type=pytorch \
  --model_type bert-mlm \
  --tokenizer_name bpe-vanilla-16k \
  --hidden_act gelu \
  --hidden_size 1024 \
  --num_hidden_layers 24 \
  --num_attention_heads 16 \
  --intermediate_size 4096 \
  --hidden_dropout_prob 0.1 \
  --attention_probs_dropout_prob 0.1 \
  --encoder_ln_mode pre-ln \
  --lr 1e-3 \
  --train_batch_size 4032 \
  --train_micro_batch_size_per_gpu 32 \
  --lr_schedule time \
  --curve linear \
  --warmup_proportion 0.06 \
  --gradient_clipping 0.0 \
  --optimizer_type adamw \
  --weight_decay 0.01 \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_eps 1e-6 \
  --total_training_time 24.0 \
  --early_exit_time_marker 24.0 \
  --dataset_path /home/olab/shakedy/academic-budget-bert/dataset/bpe16k-samples \
  --output_dir ./training_bert_bpe_16k_out \
  --print_steps 100 \
  --load_training_checkpoint /home/olab/shakedy/academic-budget-bert/training_bert_bpe_16k_out/pretrain_bert_bpe_16k- \
  --num_epochs_between_checkpoints 10000 \
  --job_name pretrain_bert_bpe_16k \
  --project_name budget-bpe-16k_pretraining \
  --validation_epochs 3 \
  --validation_epochs_begin 1 \
  --validation_epochs_end 1 \
  --validation_begin_proportion 0.05 \
  --validation_end_proportion 0.01 \
  --validation_micro_batch 16 \
  --deepspeed \
  --data_loader_type dist \
  --do_validation \
  --seed 42 \
  --fp16 \
  --finetune_checkpoint_at_end
  