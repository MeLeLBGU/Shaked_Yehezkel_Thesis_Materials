#!/bin/sh

# this is run with "sbatch conll_en_bpe_ner.sh", from the academic-budget-bert directory.

#SBATCH --job-name=conllpp_en_bpe_ner
#SBATCH --output=conllpp_en_bpe_ner_all.out
#SBATCH --error=conllpp_en_bpe_ner_all.err
#SBATCH --partition=killable
#SBATCH --nodelist=rack-omerl-g01
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=150Mib

python run_conll_ner.py \
  --model_name_or_path ./training_bert_bpe_16k_out/pretrain_bert_bpe_16k_tokenizer-/epoch1000000_step5717/  \
  --tokenizer_name bpe-vanilla-16k \
  --dataset_name conllpp \
  --text_column_name tokens \
  --label_column_name ner_tags \
  --output_dir ./conllpp/bpe_out \
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
