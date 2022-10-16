#!/bin/bash

# this is run with "sbatch generate_samples_bpe.sh" from /home/olab/shakedy/academic-budget-bert/dataset directory.

#SBATCH --job-name=generate_samples_bpe
#SBATCH --output=generate_samples_bpe.out
#SBATCH --error=generate_samples_bpe.err
#SBATCH --partition=cpu-killable

python generate_samples.py \
    --dir ../../data/wiki_shards \
    -o ../../data/bpe-16-samples \
    --dup_factor 10 \
    --vocab_file ../bpe-16k-tokenizer/vocab.txt \
    --do_lower_case 1 \
    --masked_lm_prob 0.15 \
    --max_seq_length 128  \
    --model_name bert-large-uncased \
    --max_predictions_per_seq 20 \
    --n_processes 16
