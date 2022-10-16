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

DOMAIN_TRANSFER_DIRPATH = "../../results/16k_vocab_1.9/Analysis"
DOMAIN_TRANSFER_BPE_ENCODING_EN_FILENAME = "tr_domain_transfer_bpe_encoding.txt"
DOMAIN_TRANSFER_SAGE_ENCODING_EN_FILENAME = "tr_domain_transfer_sage_encoding.txt"

##########################################################################################################################################
############################################## Contextual Exponence ######################################################################
##########################################################################################################################################

def zero_value():
    return 0

def contextual_exponence_to_file(window_size, encoding_file, freqs_out_file, normalized_freqs_out_file):
    contexts = defaultdict(set)
    token_instances = defaultdict(zero_value)
    lines = 0
    with open(encoding_file, 'r') as f:
        for l in tqdm(f):
            lines += 1
            words = l.strip().split(' ')
            for i,t in enumerate(words):
                token_instances[t] += 1
                for j in range(i-window_size, i+window_size+1):
                    if j < 0 or j == i or j >= len(words):
                        continue
                    contexts[t].add(words[j])
    print("vocab of size {}; {} lines".format(len(contexts), lines))

    # sort frequencies and write to file
    freqs = {k: len(v) for k, v in contexts.items()}
    ord_freqs = list(sorted(freqs.items(), key=lambda x: -x[1]))
    with open(freqs_out_file, 'w') as outf:
        for fr in tqdm(ord_freqs):
            outf.write("{}\n".format(fr[1]))

    # compute normalized frequencies and write to file
    normalized_freqs = {k: (float(freqs[k]) / token_instances[k]) for k, _ in freqs.items()}
    normalized_ord_freqs = list(sorted(normalized_freqs.items(), key=lambda x: -x[1]))
    with open(normalized_freqs_out_file, 'w') as noutf:
        for fr in tqdm(normalized_ord_freqs):
            noutf.write("{}\n".format(fr[1]))

    # uncomment if want to compute mean and stdev
    #mean = statistics.mean(normalized_freqs.values())
    #std = statistics.stdev(normalized_freqs.values())
    #print('Done - {} mean: {}, stdev: {}'.format(encoding_file, mean, std))
    return token_instances

def contextual_exponence_analysis():
    '''window_size = 2
    contextual_exponence_to_file(window_size, os.path.join(DIRPATH, BPE_VANILLA_ENCODING_FILENAME), os.path.join(OUTPATH, "en_bpe_vanilla_freqs_w2.txt"))
    contextual_exponence_to_file(window_size, os.path.join(DIRPATH, SAGE_ENCODING_FILENAME), os.path.join(OUTPATH, "en_sage_freqs_w2.txt"))'''

    '''window_size = 5
    contextual_exponence_to_file(window_size, os.path.join(DIRPATH, BPE_VANILLA_ENCODING_FILENAME), os.path.join(OUTPATH, "en_bpe_vanilla_freqs_w5.txt"), os.path.join(OUTPATH, "en_bpe_vanilla_normalized_freqs_w5.txt"))
    contextual_exponence_to_file(window_size, os.path.join(DIRPATH, SAGE_ENCODING_FILENAME), os.path.join(OUTPATH, "en_sage_freqs_w5.txt"), os.path.join(OUTPATH, "en_sage_normalized_freqs_w5.txt"))'''

    '''window_size = 5
    bpe_freq = contextual_exponence_to_file(window_size, os.path.join(DIRPATH, TR_BPE_VANILLA_ENCODING_FILENAME), os.path.join(OUTPATH, "tr_bpe_vanilla_freqs_w5.txt"), os.path.join(OUTPATH, "tr_bpe_vanilla_normalized_freqs_w5.txt"))
    sage_freq = contextual_exponence_to_file(window_size, os.path.join(DIRPATH, TR_SAGE_ENCODING_FILENAME), os.path.join(OUTPATH, "tr_sage_freqs_w5.txt"), os.path.join(OUTPATH, "tr_sage_normalized_freqs_w5.txt"))'''
    
    # DOMAIN TRANSFER
    window_size = 5
    bpe_freq = contextual_exponence_to_file(window_size,  
                                                os.path.join(DOMAIN_TRANSFER_DIRPATH, DOMAIN_TRANSFER_BPE_ENCODING_EN_FILENAME),
                                                os.path.join(OUTPATH, "tr_domain_transfer_bpe_freqs_w5.txt"), 
                                                os.path.join(OUTPATH, "tr_domain_transfer_bpe_normalized_freqs_w5.txt"))
    sage_freq = contextual_exponence_to_file(window_size, 
                                                os.path.join(DOMAIN_TRANSFER_DIRPATH, DOMAIN_TRANSFER_SAGE_ENCODING_EN_FILENAME), 
                                                os.path.join(OUTPATH, "tr_domain_transfer_sage_freqs_w5.txt"), 
                                                os.path.join(OUTPATH, "tr_domain_transfer_sage_normalized_freqs_w5.txt"))
    
    #bpe_freq_histo = [bpe_freq[t] for t in bpe_freq.keys()]
    #sage_freq_histo = [sage_freq[t] for t in sage_freq.keys()]

    #figure(figsize=(18, 10), dpi=80)
    #top_tokens_size = min(len(bpe_freq_histo), len(sage_freq_histo))
    
    #plt.bar(range(top_tokens_size), bpe_freq_histo[:top_tokens_size], label="BPE", alpha=0.5)
    #plt.bar(range(top_tokens_size), sage_freq_histo[:top_tokens_size], label="SAGE", alpha=0.5)

    #plt.legend()
    #plt.show()

def main():
    contextual_exponence_analysis()

if __name__ == "__main__":
    main()
