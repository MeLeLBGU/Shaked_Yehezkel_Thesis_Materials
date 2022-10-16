import os
import ast
import csv

DIRPATH = "../../results/16k_vocab_1.9/Analysis/NER"
EXPECTED_PREDICTIONS_FILENAME = "sage_expected_predictions_en_9.10.txt"
ACTUAL_PREDICTIONS_FILENAME = "sage_predictions_en_9.10.txt"
MISTAKES_FILENAME = "mistakes_sage_en_9.10.txt"

def get_mistakes_indices(mistakes_filename):
    mistakes_set = set()
    with open(os.path.join(DIRPATH, mistakes_filename), "r") as mistakes:
        mistakes_lines = mistakes.readlines()
    
    for line in mistakes_lines:
        mistake_index = int(line.split("]")[0].split("[")[1])
        mistakes_set.add(mistake_index)
    
    return mistakes_set

def mistakes_only_one_method():
    BPE_MISTAKES_FILENAME = "mistakes_bpe_en_9.10.txt"
    SAGE_MISTAKES_FILENAME = "mistakes_sage_en_9.10.txt"
    BPE_ONLY_MISTAKES_FILENAME = "mistakes_bpe_only_en_9.10.txt"
    SAGE_ONLY_MISTAKES_FILENAME = "mistakes_sage_only_en_9.10.txt"

    bpe_mistakes = sorted(get_mistakes_indices(BPE_MISTAKES_FILENAME))
    sage_mistakes = sorted(get_mistakes_indices(SAGE_MISTAKES_FILENAME))

    bpe_only_mistakes = [i for i in bpe_mistakes if i not in sage_mistakes]
    sage_only_mistakes = [i for i in sage_mistakes if i not in bpe_mistakes]
    
    '''with open(os.path.join(DIRPATH, BPE_ONLY_MISTAKES_FILENAME), "w") as bpe_only:
        for m in bpe_only_mistakes:
            bpe_only.write("{}\n".format(m))'''

    '''with open(os.path.join(DIRPATH, SAGE_ONLY_MISTAKES_FILENAME), "w") as sage_only:
        for m in sage_only_mistakes:
            sage_only.write("{}\n".format(m))'''

    return bpe_only_mistakes, sage_only_mistakes

def mistakes_to_csv():
    '''
    word,true_pred,bpe_pred,sage_pred
    ...
    new line between new sentences
    '''
    EXPECTED_PREDICTIONS_FILENAME = "expected_predictions_conll.txt"
    SAGE_ACTUAL_PREDICTIONS_FILENAME = "sage_predictions_conll.txt"
    BPE_ACTUAL_PREDICTIONS_FILENAME = "bpe_predictions_conll.txt"
    BPE_ANALYSIS_FILENAME = "conll_bpe_only_mistakes.csv"
    SAGE_ANALYSIS_FILENAME = "conll_sage_only_mistakes.csv"

    #print("[*] Computing mistakes for each method")
    #bpe_only_mistakes, sage_only_mistakes = mistakes_only_one_method()

    print("[*] Opening files")
    expected_predictions = open(os.path.join(DIRPATH, EXPECTED_PREDICTIONS_FILENAME), "r")
    sage_actual_predictions = open(os.path.join(DIRPATH, SAGE_ACTUAL_PREDICTIONS_FILENAME), "r")
    bpe_actual_predictions = open(os.path.join(DIRPATH, BPE_ACTUAL_PREDICTIONS_FILENAME), "r")

    sage_analysis_file = open(os.path.join(DIRPATH, SAGE_ANALYSIS_FILENAME), "w")
    bpe_analysis_file = open(os.path.join(DIRPATH, BPE_ANALYSIS_FILENAME), "w")

    sage_analysis_writer = csv.writer(sage_analysis_file)
    bpe_analysis_writer = csv.writer(bpe_analysis_file)

    sage_analysis_writer.writerow(["word", "true prediction", "bpe", "sage"])
    bpe_analysis_writer.writerow(["word", "true prediction", "bpe", "sage"])
    
    print("[*] Starting loop")
    expected_lines = expected_predictions.readlines()
    len_expected = len(expected_lines)
    written = 0
    for i, expected in enumerate(expected_lines):
        if i%500 == 0:
            print("[*] processing line {}/{}".format(i, len_expected))

        ## sentence
        true_preds = ast.literal_eval(expected.split("\n")[0].split("ner: ")[1])
        words = ast.literal_eval(expected.split("words: ")[1].split(", ner: ")[0])

        bpe_preds = bpe_actual_predictions.readline().split("\n")[0].split(" ")
        sage_preds = sage_actual_predictions.readline().split("\n")[0].split(" ")

        sentence_data = []
        sage_mistake = False
        bpe_mistake = False
        for w, true_pred, bpe_pred, sage_pred in zip(words, true_preds, bpe_preds, sage_preds):
            ## this should be line in analysis file
            if w == "" or w == "," or w == "":
                continue
            sentence_data += [w, true_pred, bpe_pred, sage_pred]
            if bpe_pred != true_pred:
                bpe_mistake = True
            if sage_pred != true_pred:
                sage_mistake = True

        if sage_mistake != bpe_mistake:
            if sage_mistake:
                print("[*] sage mistake...")
                sage_analysis_writer.writerow(sentence_data)
                sage_analysis_writer.writerow([])
                sage_analysis_writer.writerow([])
            else:
                print("[*] bpe mistake...")
                bpe_analysis_writer.writerow(sentence_data)
                bpe_analysis_writer.writerow([])
                bpe_analysis_writer.writerow([])
            written += 1

    expected_predictions.close()
    sage_actual_predictions.close()
    bpe_actual_predictions.close()
    sage_analysis_file.close()
    bpe_analysis_file.close()

def main():
    expected_preds = open(os.path.join(DIRPATH, EXPECTED_PREDICTIONS_FILENAME), "r")
    actual_preds = open(os.path.join(DIRPATH, ACTUAL_PREDICTIONS_FILENAME), "r")
    mistakes = open(os.path.join(DIRPATH, MISTAKES_FILENAME), "w")

    for i, expected in enumerate(expected_preds.readlines()):
        expected_ner = ast.literal_eval(expected.split("\n")[0].split("ner: ")[1])
        words = expected.split("words: ")[1].split(", ner: ")[0]
        #print(expected_ner)
        #print(words)
        # actual prediction for that index
        actual = actual_preds.readline()
        actual_ner = actual.split("\n")[0].split(" ")
        #print(actual_ner)
        #print(expected_ner == actual_ner)
        if expected_ner != actual_ner:
            mistakes.write("[{}]: words: {}, prediction: {}\n".format(i, words, actual_ner))

if __name__ == "__main__":
    #main()
    #mistakes_only_one_method()
    mistakes_to_csv()
