## Execute from tokenization directory in mac ##

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import statistics
import json
import numpy as np
import ast
import os

DIRPATH = "../../results/16k_vocab_1.9/create_16k_vocab_1.9/"
BPE_VANILLA_VOCAB_FILENAME = "bpe_vanilla_500_vocab.txt"
SAGE_VOCAB_FILENAME = "sg_bpe_500_vocab.txt"
INITIAL_BPE_VOCAB_FILENAME = "sg_bpe_original_550_vocab.txt"
BPE_VANILLA_VOCAB_ONLY_FILENAME = "bpe_vanilla_500_only.txt"
SAGE_VOCAB_ONLY_FILENAME = "sg_bpe_500_only.txt"
BPE_VANILLA_ENCODING_FILENAME = "bpe_vanilla_500_encoding.txt"
INITIAL_BPE_VANILLA_ENCODING_FILENAME = "sg_bpe_original_550_encoding.txt"
SAGE_ENCODING_FILENAME = "sg_bpe_500_encoding.txt"

TR_BPE_VANILLA_VOCAB_FILENAME = "tr_bpe_vanilla_500_vocab.txt"
TR_SAGE_VOCAB_FILENAME = "tr_sg_bpe_500_vocab.txt"
TR_BPE_VANILLA_VOCAB_ONLY_FILENAME = "tr_bpe_vanilla_500_only.txt"
TR_SAGE_VOCAB_ONLY_FILENAME = "tr_sg_bpe_500_only.txt"
TR_BPE_VANILLA_ENCODING_FILENAME = "tr_bpe_vanilla_500_encoding.txt"
TR_SAGE_ENCODING_FILENAME = "tr_sg_bpe_500_encoding.txt"

OUTPATH = "../../results/16k_vocab_1.9/Analysis"

##########################################################################################################################################
############################################## Frequency #################################################################################
##########################################################################################################################################

NEW_WORD_SEPARATOR = "\xe2\x96\x81"

def get_encoding_tokens(encoding_filepath):
    with open(encoding_filepath) as encoding_file:
        data = encoding_file.read()
    
    tokens_in_data = data.split(" ")
    for i, t in enumerate(tokens_in_data):
        if t.startswith(NEW_WORD_SEPARATOR):
            tokens_in_data[i] = "_" + t[len(NEW_WORD_SEPARATOR):]
        tokens_in_data[i] = tokens_in_data[i].lower()

    return tokens_in_data

def frequency_in_vocab(vocab_filepath, vocab_only_filepath, encoding_filepath):
    with open(vocab_filepath, "r") as vocab_file:
        vocab_data_str = vocab_file.read()
    # the file is saved as list string, convert to python list
    vocab_list = ast.literal_eval(vocab_data_str)

    with open(vocab_only_filepath, "r") as vocab_only_file:
        vocab_only_data_str = vocab_only_file.read()
    vocab_only_list = ast.literal_eval(vocab_only_data_str)

    ## We want frequency for tokens found in both vocabularies
    shared_vocab_list = list(set(vocab_list) - set(vocab_only_list))

    ## Now compute frequency for each v in shared_vocab_list
    tokens_in_data = get_encoding_tokens(encoding_filepath)
    print("Got encoding tokens")
    
    frequency_dict = {}
    for i, t in enumerate(shared_vocab_list):
        if (i%100 == 0):
            print("[{}] {} / {}".format(datetime.now().time(), i, len(shared_vocab_list)))
        if t.startswith(NEW_WORD_SEPARATOR):
            t =  "_" + t[len(NEW_WORD_SEPARATOR):]
        t = t.lower()

        frequency_dict[t] = tokens_in_data.count(t)

    return frequency_dict

def frequency_analysis():
    print("Computing frequency in sage")
    sage_frequency = frequency_in_vocab(os.path.join(DIRPATH, SAGE_VOCAB_FILENAME),
                                        os.path.join(DIRPATH, SAGE_VOCAB_ONLY_FILENAME),
                                        os.path.join(DIRPATH, SAGE_ENCODING_FILENAME))

    print("Writing")
    with open(os.path.join(OUTPATH, "sage_freq.txt"), "w+") as sorted_freq:
        sorted_freq.write(json.dumps(sage_frequency, indent=4))

    print("Computing frequency in bpe")
    bpe_frequency = frequency_in_vocab(os.path.join(DIRPATH, BPE_VANILLA_VOCAB_FILENAME),
                                       os.path.join(DIRPATH, BPE_VANILLA_VOCAB_ONLY_FILENAME),
                                       os.path.join(DIRPATH, BPE_VANILLA_ENCODING_FILENAME))

    with open(os.path.join(OUTPATH, "bpe_freq.txt"), "w+") as sorted_freq:
        sorted_freq.write(json.dumps(bpe_frequency, indent=4))

    print("Computing differences")
    differences_frequency = {key: abs(sage_frequency[key] - bpe_frequency[key]) for key in sage_frequency}
    differences_freq_sorted = sorted(differences_frequency.iterkeys(), key=lambda k: differences_frequency[k], reverse=True)
    
    print("Writing differences")
    with open(os.path.join(OUTPATH, "sorted_freq.txt"), "w+") as sorted_freq:
        sorted_freq.write("\n".join(differences_freq_sorted))

def frequency_from_files():
    with open(os.path.join(OUTPATH, "sage_freq.txt"), "r") as sage_freq:
        sage_freq = json.loads(sage_freq.read())

    with open(os.path.join(OUTPATH, "bpe_freq.txt"), "r") as bpe_freq:
        bpe_freq = json.loads(bpe_freq.read())

    sage_freq_diff = {key: (sage_freq[key] - bpe_freq[key]) for key in sage_freq}
    sage_freq_diff_sorted = sorted(sage_freq_diff.iterkeys(), key=lambda k: sage_freq_diff[k], reverse=True)
    with open(os.path.join(OUTPATH, "sage_freq_sorted.txt"), "w+") as sorted_sage_freq:
        sorted_sage_freq.write(json.dumps(sage_freq_diff_sorted, indent=4))

    bpe_freq_diff = {key: (bpe_freq[key] - sage_freq[key]) for key in bpe_freq}
    bpe_freq_diff_sorted = sorted(bpe_freq_diff.iterkeys(), key=lambda k: bpe_freq_diff[k], reverse=True)
    with open(os.path.join(OUTPATH, "bpe_freq_sorted.txt"), "w+") as sorted_bpe_freq:
        sorted_bpe_freq.write(json.dumps(bpe_freq_diff_sorted, indent=4))

def main():
    frequency_analysis()
    #frequency_from_files()
    
if __name__ == "__main__":
    main()
