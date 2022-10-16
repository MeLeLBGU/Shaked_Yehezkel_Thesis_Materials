#!/bin/bash

# this is run with "sbatch make_shards.sh" from academic-budget-bert/dataset directory.

#SBATCH --job-name=make_shards_ru
#SBATCH --output=make_shards_ru.out
#SBATCH --error=make_shards_ru.err
#SBATCH --partition=cpu-killable

python shard_data.py \
    --dir ../../data/wiki_different_languages/ru \
    -o ../../data/ru_wiki_shards \
    --num_train_shards 256 \
    --num_test_shards 128 \
    --frac_test 0.1
