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

#DIRPATH = "../../results/16k_vocab_1.9/create_16k_vocab_1.9/"
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

#DOMAIN_TRANSFER_
DIRPATH = "../../results/16k_vocab_1.9/Analysis"
DOMAIN_TRANSFER_BPE_ENCODING_FILENAME = "domain_transfer_bpe_encoding.txt"
DOMAIN_TRANSFER_SAGE_ENCODING_FILENAME = "domain_transfer_sage_encoding.txt"

OUTPATH = "../../results/16k_vocab_1.9/Analysis"

##########################################################################################################################################
############################################## Fertility #################################################################################
##########################################################################################################################################

NEW_WORD_SEPARATOR = "\xe2\x96\x81"
NEW_TOKEN_SEPARATOR = " "
def get_tokens_per_words_list(encoding_filepath):
    ## we should take the encoding, split by start of a word ("_"), and get the number of tokens (split by " ") that we got there
    ## then plot histogram...
    with open(encoding_filepath) as encoding_file:
        data = encoding_file.read()
    
    words_in_data = data.split(NEW_WORD_SEPARATOR)
    tokens_per_words = [w.count(NEW_TOKEN_SEPARATOR) for w in words_in_data]
    return tokens_per_words

def plot_fertility_hist(first_encoding_filepath, first_encoding_title, first_encoding_color, 
                        second_encoding_filepath, second_encoding_title, second_encoding_color):
    first_encoding_fertility = get_tokens_per_words_list(os.path.join(DIRPATH, first_encoding_filepath))
    second_encoding_fertility = get_tokens_per_words_list(os.path.join(DIRPATH, second_encoding_filepath))

    plt.hist([first_encoding_fertility, second_encoding_fertility], \
        color=[first_encoding_color, second_encoding_color], \
        label=[first_encoding_title, second_encoding_title], bins=range(9))
    ticks = [(t + 0.5) for t in range(8)][1:]
    ticklabels = range(8)[1:]
    plt.xlim(left=0.5)
    plt.xlabel("Subwords in word")
    plt.ylabel("Words in corpus")
    plt.legend()
    plt.xticks(ticks, ticklabels)
    plt.yticks([1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000], ["1M", "2M", "3M", "4M", "5M", "6M", "7M"])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPATH, "fertility_histogram_transfer_en_{}_{}.png".format(first_encoding_title, second_encoding_title)))
    plt.show()

'''
        ticks = [(2 * t + 0.5) for t in range(8)]
    ticklabels = [2 * t for t in range(16)]
    plt.ylim(0, 4000)
    plt.xlim(left=0.5)
    plt.xlabel("Token Length")
    plt.legend()
    plt.xticks(ticks, ticklabels)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPATH, "lengths_histogram_{}_{}.png".format(first_vocab_title, second_vocab_title)))
    plt.show()
'''

def fertility_analysis():
    #plot_fertility_hist(BPE_VANILLA_ENCODING_FILENAME, "transfer_BPE", "darkorange", SAGE_ENCODING_FILENAME, "SaGe", "darkgreen")
    #plot_fertility_hist(INITIAL_BPE_VANILLA_ENCODING_FILENAME, "bpe_20k", "green", SAGE_ENCODING_FILENAME, "sage_16k", "darkorange")
    plot_fertility_hist(DOMAIN_TRANSFER_BPE_ENCODING_FILENAME, "BPE", "darkorange", DOMAIN_TRANSFER_SAGE_ENCODING_FILENAME, "SaGe", "darkgreen")
    
def main():
    fertility_analysis()
    
if __name__ == "__main__":
    main()
