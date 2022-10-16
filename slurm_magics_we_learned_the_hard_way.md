## Auto-Restart:
----------------
Slurm can cancel your job whenever it wants.
I wanted to have automatic mechanism to restart the job from last checkpoint.
For that I used slurm "--signal" parameter to make it send signal to my job just before it ends it,
and also the "trap" linux command, that defines what to do whenever we get that signal.
Then, I created 2 batch files - "stage1" and "stage2", where "stage1" defined used "--signal" and "trap" to start "stage2" on stop, 
and "stage2" had the tailored command to start over from checkpoint.
See example for vocabulary creation script in later section.

## Restrictions:
----------------
It turns out that some nodes in slurm are better than others, in term of memory consumption, gpu types, etc.
You can specify nodelist of nodes you want to execute on, or how many memory you want..
For example, this is the combination that worked for me (on TAU slurm)-
```
#SBATCH --job-name=conll_en_bpe_ner
#SBATCH --output=conll_en_bpe_ner.out
#SBATCH --error=conll_en_bpe_ner.err
#SBATCH --partition=killable
#SBATCH --nodelist=rack-omerl-g01
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=150Mib
```
(killable is the partition with GPUs, cpu-killable is CPU-only)

## Other:
----------
1. You can use %j in the script to embed the jobid (for example to get another name for output\error file).

## Practical Examples:
----------------------

#### Example of batch files for vocabulary creation:

** stage 1 **
```
#!/bin/bash

# this is run with "sbatch create_16k_vocab_stage1.sh

#SBATCH --job-name=create_16k_vocab_stage1
#SBATCH --output=create_16k_vocab_stage1.out
#SBATCH --error=create_16k_vocab_stage1.err
#SBATCH --partition=killable

# asks SLURM to send the USR1 signal 10 seconds before end of the time limit
#SBATCH --signal=B:USR1@10

# define the handler function - this is executed when the signal is sent
cleanup_function()
{
    echo "starting stage2... at $(date)" >> create_16k_vocab_stage1.out
    sbatch create_16k_vocab_stage2.sh
}

# call cleanup_function once we receive USR1 signal
trap 'cleanup_function' USR1

# ---------------------------------------------------------------------------------
echo "starting calculation at $(date)" > create_16k_vocab_stage1.out
rm -rf ./results/create_16k_vocab
mkdir ./results/create_16k_vocab
python Main.py create_16k_vocab --final_vocab_size 16000 --initial_vocab_size 20000 --tokens_to_prune_in_iteration 100 --tokens_to_consider_in_iteration 1500 --iterations_until_reranking 10 --corpus_filepath "../data/wiki_lines.txt" --partial_corpus_filepath "../data/wiki_lines_partial.txt" --thousands_of_corpus_lines 750 --use_gensim Y &
# ---------------------------------------------------------------------------------
```

** stage 2 **
```
#!/bin/bash

# this is run with "sbatch create_16k_vocab_stage2.sh from trap function

#SBATCH --job-name=create_16k_vocab_stage2
#SBATCH --output=create_16k_vocab_stage2.%j.out
#SBATCH --error=create_16k_vocab_stage2.%j.err
#SBATCH --partition=killable

# asks SLURM to send the USR1 signal 10 seconds before end of the time limit
#SBATCH --signal=B:USR1@10

# define the handler function - this is executed when the signal is sent
cleanup_function()
{
    echo "starting stage2... at $(date)" >> create_16k_vocab_stage2.%j.out
    sbatch create_16k_vocab_stage2.sh
}

# call cleanup_function once we receive USR1 signal
trap 'cleanup_function' USR1

# ---------------------------------------------------------------------------------
echo "starting calculation at $(date)" > create_16k_vocab_stage2.out
python Main.py create_16k_vocab --final_vocab_size 16000 --initial_vocab_size 20000 --tokens_to_prune_in_iteration 100 --tokens_to_consider_in_iteration 1500 --iterations_until_reranking 10 --corpus_filepath "../data/wiki_lines.txt" --partial_corpus_filepath "../data/wiki_lines_partial.txt" --thousands_of_corpus_lines 750 --use_gensim Y --is_continue Y &
# ---------------------------------------------------------------------------------
```

#### Example of batch file for bert pretraining:
```
#!/bin/sh

# this is run with "sbatch pretrain_bert.sh", from the academic-budget-bert directory.

#SBATCH --job-name=pretrain_bert
#SBATCH --output=pretrain_bert.out
#SBATCH --error=pretrain_bert.err
#SBATCH --partition=killable
#SBATCH --gpus=3
#SBATCH --mem=50000

deepspeed run_pretraining.py \
  --layer_norm_type=pytorch \
  --model_type bert-mlm \
  --tokenizer_name <tokenizer dirname> \
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
  --total_training_time <how many time you want it to train> \
  --early_exit_time_marker <how many time you want it to train> \
  --dataset_path <samples-dirpath>  \
  --output_dir <out-path> \
  --print_steps 100 \
  --num_epochs_between_checkpoints 10000 \
  --job <jobname> \
  --project_name <for wandb> \
  --validation_epochs 3 \
  --validation_epochs_begin 1 \
  --validation_epochs_end 1 \
  --validation_begin_proportion 0.05 \
  --validation_end_proportion 0.01 \
  --validation_micro_batch 16 \
  --deepspeed \
  --data_loader_type dist \
  --do_validation \
  --fp16 \
  --finetune_checkpoint_at_end
```

run_pretraining can load from last checkpoint.
To do this, add: ```--load_training_checkpoint <path>``` (with the correct path).

#### Example of batch file for fine tuning:
```
#!/bin/sh

# this is run with "sbatch <filename>.sh", from the academic-budget-bert directory.

#SBATCH --job-name=<filename>
#SBATCH --output=<XXX>.%j.out
#SBATCH --error=<XXX>.%j.err
#SBATCH --partition=killable
#SBATCH --gpus=3
#SBATCH --mem=50000

python run_glue.py \
  --model_name_or_path <path> \
  --tokenizer_name <tokenizer-path> \
  --task_name WNLI \
  --max_seq_length 128 \
  --output_dir <out-path> \
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
```
