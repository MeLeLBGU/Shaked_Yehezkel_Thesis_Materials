In order to pretrain bert model for your created vocabulary, do the following:
1. Do the "one-time" preparations.
2. Tailor your tokenizer to the bert24 code.
3. Prepare dataset.
4. Create dataset samples using your tailored tokenizer.
5. Execute run_pretraining.py from academic-budget-bert.
6. Continue from checkpoint 
See instructions for how to do that in the following sections.

# 1) One-time preparations:
----------------------------
In order to pretrain bert (and later finetuning it), we chose to use the "academic-budget-bert" code (https://arxiv.org/abs/2104.07705).

1. Clone https://github.com/IntelLabs/academic-budget-bert
2. Make sure you have python 3.6+, apex, and pytorch.
3. Install the requirements (pip install -r requirements.txt) - see "Installation" section in academic-budget-bert readme.

# 2) How to create tokenizer for bert24 code?
----------------------------------------------
After we have a vocab file, we should have it in the huggingface format so we can pass it as input to Bert24 pretraining.

1. Create new folder, with the name of your tokenizer (e.g. “sage-tokenizer” or “bpe-tokenizer).
2. We get the generated <xxx_vocab.txt> file from the vocab generation script (it will be in results directory under the experiment-name directory).
3. Execute utils\convert_vocab_for_hf_tokenizer.py (from this repo) to get the converted vocab file. Save it as vocab.txt in your new tokenizer folder created in [1].
4. Add the following tokens to the top of converted vocab file:
[PAD]
[UNK]
[CLS]
[SEP]
[MASK]
5. Create in your tokenizer folder “tokenizer_config.json”
    a. You can copy the one of sg-tokenizer-8k and change the following:
        i. Change “name_or_path”
        ii. Change any other parameter that changed
6. Copy “tokenizer_config.json” to “config.json” in your tokenizer folder (have them both).
7. Copy “special_tokens_map.json” from sg-tokenizer-8k to your new tokenizer folder.

Note that after you have a newly tokenizer and vocab file, you should create new samples (dataset) using your tokenizer!
(You must create new samples for every new tokenizer!)

# 3) Prepare dataset:
---------------------
Refers to: https://github.com/IntelLabs/academic-budget-bert/tree/04f6da685acf4dfc47b85b42307e17340e87fde3/dataset

1. Process_data.py: SHOULD BE DONE ONCE-PER-DATASET

    a. You probably already done this step when preparing data for vocab_creation script.

    b. The process_data.py script takes wikipedia XML, and creates one united file - wiki_one_article_per_line.txt.

    c. The script does it by calling wikiextractor.WikiExtractor python module 

        i. (https://github.com/attardi/wikiextractor)

    d. For your help - the git of WikiExtractor references path to the latest wiki dump 

        i. (https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

2. Shard_data.py: SHOULD BE DONE ONCE-PER-DATASET

    a. Creates some shards from the united huge file created before.

    b. It first uses NTLKSegmenter to segment the big file for sentences (NLTK can tokenize to words / sentences. Here they use nltk.tokenizer.sent_tokenize() that tokenize to sentences).

    c. After that it loads articles (lines) into dictionary of articles, segment each article to sentences, and randomly divide them to shards. Finally writes them to disk.

# 4) How to create samples using your tokenizer?
-------------------------------------------------
- generate_samples.py, should be done ONCE-PER-**TOKENIZER**.

    a. Creates hdf5 files.

    b. It uses create_pretraining_data.py for that (if its not roberta).

    c. Initializes BertTokenizer with the vocab file given. Uses create_training_instances (that gets tokenizer). Create_training_instances opens a shard file, reads line and tokenize it, and put in some sample. This script also creates the predictions - and for some percent of predictions it create negative Masked LM prediction - using a random token from the given vocabulary.

1. Assume your shards are at /data/out
2. Execute:
```
python generate_samples.py --dir ../data/out/ -o ../data/samples_bpe16k --dup_factor 10 --vocab_file ../bpe-vanilla-16k/vocab.txt --do_lower_case 1 --masked_lm_prob 0.15 --max_seq_length 128  --model_name bert-large-uncased --max_predictions_per_seq 20 --n_processes 16 &
```
3. The path to the just-created samples should be given to run_pretraining and the finetuning scripts.

# 5) How to pretrain bert24?
-----------------------------
1. Make sure you have the correct tokenizer (look at “2) How to create tokenizer for bert24 code?”).
2. Upload your tokenizer folder to the server, I used to put it under the “academic-budget-bert” directory just cloned.
3. Change “pretrain_bert.sh” according to your parameters. Notice especially:

    a. Change name of job, output/error file according to your execution.

    b. --tokenizer_name <your-tokenizer-folder-name-here>
    
    c. Give the right dataset samples path - For example, /home/olab/shakedy/academic-budget-bert/data/samples_bpe16k
        Make sure you created samples suit to your vocab file! 

    d. Give new name to –output_dir

4. Upload to slurm, cd to /home/olab/shakedy/academic-budget-bert, and “sbatch pretrain_bert.sh”.

    a. the "run_pretraining.py" should be executed from the academic-budget-bert directory!

5. After 24 hours (or how many hours you wanted) you should have the first checkpoint in your output dir you chose.

# 6) How to continue after 24 hours?
-----------------------------
You probably now have the checkpoint under the output dir you chose.
Slurm ends just a little before 24 hours for a job, so it probably did not end the 24 hours pretraining….

a. Change pretrain_bert_from_checkpoint.sh to continue from your checkpoint.

b. Make sure to change relevant parameters in the bash script

c. Execute sbatch pretrain_bert_from_checkpoint.sh, until the training ends.
