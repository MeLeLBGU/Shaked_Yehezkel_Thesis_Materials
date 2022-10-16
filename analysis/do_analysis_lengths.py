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

plt.rcParams.update({'font.size': 16})

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
############################################## Lengths ###################################################################################
##########################################################################################################################################

def get_vocab_lengths(vocab_filepath):
    with open(vocab_filepath, "r") as vocab_file:
        vocab_data_str = vocab_file.read()
    
    # the file is saved as list string, convert to python list
    vocab_list = ast.literal_eval(vocab_data_str)

    # undo preprocess of \u2581
    vocab_list = [t if not ("\\u2581" in t) else (t.split("\\u2581")[1]) for t in vocab_list]

    # remove special tokens
    if "<unk>" in vocab_list:
        vocab_list.remove("<unk>")
    if "<s>" in vocab_list:
        vocab_list.remove("<s>")
    if "</s>" in vocab_list:
        vocab_list.remove("</s>")

    # vocab lengths
    vocab_lengths_list = [len(t) for t in vocab_list]
    return vocab_lengths_list, len(set(vocab_lengths_list))

def plot_length_hist(first_vocab_filepath, first_vocab_title, first_vocab_color, second_vocab_filepath, second_vocab_title, second_vocab_color):
    first_vocab_lengths, _ = get_vocab_lengths(os.path.join(DIRPATH, first_vocab_filepath))
    second_vocab_lengths, _ = get_vocab_lengths(os.path.join(DIRPATH, second_vocab_filepath))

    #first_vocab_length_2 = first_vocab_lengths.count(2)
    #first_vocab_length_3 = first_vocab_lengths.count(3)
    '''print("{} length 2: {}, bpe length 3: {}, bpe all: {}".format(first_vocab_filepath, 
                                                                    first_vocab_length_2, 
                                                                    first_vocab_length_3,
                                                                    len(first_vocab_lengths)))'''
    #second_vocab_more_than_5 = [i for i in second_vocab_lengths if i > 5]
    #print("{} more than 5: {}, all: {}".format(second_vocab_filepath, len(second_vocab_more_than_5), len(second_vocab_lengths)))

    plt.hist([first_vocab_lengths, second_vocab_lengths], \
        color=[first_vocab_color, second_vocab_color], \
        label=[first_vocab_title, second_vocab_title], bins=range(16))
    ticks = [(2 * t + 0.5) for t in range(8)]
    ticklabels = [2 * t for t in range(16)]
    plt.ylim(0, 3500)
    plt.xlim(left=0.5)
    plt.xlabel("Token Length")
    plt.legend()
    plt.xticks(ticks, ticklabels)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPATH, "lengths_histogram_{}_{}.png".format(first_vocab_title, second_vocab_title)))
    plt.show()

def token_length_analysis():
    plot_length_hist(BPE_VANILLA_VOCAB_FILENAME, "BPE", "darkorange", SAGE_VOCAB_FILENAME, "SaGe", "darkgreen")
    #plot_length_hist(INITIAL_BPE_VOCAB_FILENAME, "bpe_20k", "green", SAGE_VOCAB_FILENAME, "sage_16k", "darkorange")
    #plot_length_hist(BPE_VANILLA_VOCAB_ONLY_FILENAME, "bpe_16k_only", "skyblue", SAGE_VOCAB_ONLY_FILENAME, "sage_16k_only", "orange")

def numbers_for_word_initials(vocab_filepath):
    with open(vocab_filepath, "r") as vocab_file:
        vocab_data_str = vocab_file.read()
    
    # the file is saved as list string, convert to python list
    vocab_list = ast.literal_eval(vocab_data_str)

    # undo preprocess of \u2581
    vocab_word_initials = [t for t in vocab_list if t.startswith("\\u2581")]
    #print(vocab_word_initials[:15])

    print("[*][{}] num of initials: {}, num of vocab: {}, fraction: {}".format(vocab_filepath,
                                                                                len(vocab_word_initials),
                                                                                len(vocab_list),
                                                                                float(len(vocab_word_initials))/len(vocab_list)))

def main():
    #token_length_analysis()
    numbers_for_word_initials(os.path.join(DIRPATH, SAGE_VOCAB_ONLY_FILENAME))
    numbers_for_word_initials(os.path.join(DIRPATH, BPE_VANILLA_VOCAB_ONLY_FILENAME))

if __name__ == "__main__":
    main()
