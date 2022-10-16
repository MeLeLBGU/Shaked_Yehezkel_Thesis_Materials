We wanted to add some manual analysis of the vocabularies, inspired by bostrom & durrett (https://aclanthology.org/2020.findings-emnlp.414/).

Change hardcoded filepaths before you use.

# do_analysis_exponence:
How many different neighbors one token sees.
This script outputs 
In order to plot the histogram or boxplot, use "draw_exponence_histo.py" (also as colab - https://colab.research.google.com/drive/1q77mSbTuzvBewvich37StIf0o3A4V4HP#scrollTo=FO5938e6wi-K).

# do_analysis_fertility:
Histogram for how many tokens required to tokenize a word in the given corpus.

# do_analysis_frequency:
How many times each token appears in the corpus.

# do_analysis_lengths:
Distribution for token lengths over the vocabulary.

# new_mistakes_from_files:
Some helpers to extract in readable way the NER mistakes (in the format of word,correct-prediction,bpe-prediction,sage-prediction, separated by sentences).

# encode_domain_transfer:
This script should be executed from the "vocab_creation" directory, since it uses some of its python modules.
Encodes other domain's corpus, using the output vocabulary.
We used it to output encoding files, and then used the encodings as inputs for other analysis scripts in order to test domain transferability of our method.
