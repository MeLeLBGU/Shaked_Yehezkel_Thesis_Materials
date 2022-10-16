#!/bin/bash

# this is run with "sbatch preprocess_data.sh" from academic-budget-bert/dataset directory.

#SBATCH --job-name=preprocess_data
#SBATCH --output=preprocess_data_ru.out
#SBATCH --error=preprocess_data_ru.err
#SBATCH --partition=cpu-killable

python process_data.py -f ../../data/wiki_different_languages/ru/ruwiki-latest-pages-articles.xml -o ../../data/wiki_different_languages/ru --type wiki
