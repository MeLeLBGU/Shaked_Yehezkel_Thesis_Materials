## should be executed from within "vocab_creation" directory in server.

import os
import logging
import datasets
import sentencepiece as spm

OUTPATH = "./16k_vocab_1.9_analysis/turkish"
SAGE_TRANSFER_ENCODING_FILENAME = "tr_sage_encoding_transfer.txt"
BPE_TRANSFER_ENCODING_FILENAME = "tr_bpe_encoding_transfer.txt"
MAX_WORDS_IN_CORPUS = 750 * 1000 * 10 # ~ 750K lines of 10 words

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# should take the model file path from vocabulary creation outpath
def encode_english_domain_transfer(logger, sage_model_filepath, bpe_model_filepath):
    # if you change the second param to "all" you get a huge dataset.
    logger.info("[*] loading dataset")
    english_dataset = datasets.load_dataset("pile-of-law/pile-of-law", 'us_bills', split="train")  

    logger.info("[*] loading sage model")
    sage_model = spm.SentencePieceProcessor(model_file=sage_model_filepath)
    sage_encoding_file = open(os.path.join(OUTPATH, SAGE_TRANSFER_ENCODING_FILENAME), "w")

    logger.info("[*] loading bpe model")
    bpe_model = spm.SentencePieceProcessor(model_file=bpe_model_filepath)
    bpe_encoding_file = open(os.path.join(OUTPATH, BPE_TRANSFER_ENCODING_FILENAME), "w")

    logger.info("[*] going through instances...")
    len_instances = len(english_dataset)
    words = 0
    for i, instance in enumerate(english_dataset):
        if i%500 == 0:
            logger.info("[*] instance #{}/{}".format(i, len_instances))

        instance_lines = instance["text"].strip().split("\n")
        #logger.info("instance: {}".format(instance_lines))
        for line in instance_lines:
            # encode each line and write tokenized to some outfile
            line = line.lower()
            sage_encoded_line = [sage_model.id_to_piece(x) for x in sage_model.encode(line)]
            sage_encoding_file.write(" ".join(sage_encoded_line) + " ")

            bpe_encoded_line = [bpe_model.id_to_piece(x) for x in bpe_model.encode(line)]
            bpe_encoding_file.write(" ".join(bpe_encoded_line) + " ")

            # do we need to stop?
            words += len(line.split(" "))
            if words >= MAX_WORDS_IN_CORPUS:
                sage_encoding_file.close()
                bpe_encoding_file.close()
                return

# should take the model file path from vocabulary creation outpath
def encode_turkish_domain_transfer(logger, sage_model_filepath, bpe_model_filepath):
    # if you change the second param to "all" you get a huge dataset.
    logger.info("[*] loading dataset")
    turkish_dataset = datasets.load_dataset("cansen88/turkishReviews_5_topic", split="train")  

    logger.info("[*] loading sage model")
    sage_model = spm.SentencePieceProcessor(model_file=sage_model_filepath)
    sage_encoding_file = open(os.path.join(OUTPATH, SAGE_TRANSFER_ENCODING_FILENAME), "w")

    logger.info("[*] loading bpe model")
    bpe_model = spm.SentencePieceProcessor(model_file=bpe_model_filepath)
    bpe_encoding_file = open(os.path.join(OUTPATH, BPE_TRANSFER_ENCODING_FILENAME), "w")

    logger.info("[*] going through instances...")
    len_instances = len(turkish_dataset)
    for i, instance in enumerate(turkish_dataset):
        if i%500 == 0:
            logger.info("[*] instance #{}/{}".format(i, len_instances))

        instance_text = instance['review'].lower()

        sage_encoded_text = [sage_model.id_to_piece(x) for x in sage_model.encode(instance_text)]
        sage_encoding_file.write(" ".join(sage_encoded_text))

        bpe_encoded_text = [bpe_model.id_to_piece(x) for x in bpe_model.encode(instance_text)]
        bpe_encoding_file.write(" ".join(bpe_encoded_text))

def main():
    logger = logging.getLogger("domaintransfer")

    '''sage_filepath_model = "./results/create_16k_vocab_1.9/sg_bpe.model"
    bpe_filepath_model = "./results/create_16k_vocab_1.9/bpe_vanilla.model"
    encode_english_domain_transfer(logger, sage_filepath_model, bpe_filepath_model)'''

    sage_filepath_model = "./results/create_16k_vocab_tr_19.8/sg_bpe.model"
    bpe_filepath_model = "./results/create_16k_vocab_tr_19.8/bpe_vanilla.model"
    encode_turkish_domain_transfer(logger, sage_filepath_model, bpe_filepath_model)
    
if __name__ == "__main__":
    main()
