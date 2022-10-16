''' 
This script should be executed from academic-budget-bert\dataset directory, since it uses TextSharding.
Its purpose is to split "wiki_one_article_per_line.txt" into shorter lines.
'''

from data import TextSharding

WIKI_ONE_ARTICLE_PER_LINE_FILEPATH = "/home/olab/shakedy/data/wiki_different_languages/ru/wiki_one_article_per_line.txt"
WIKI_SHORT_LINES_OUTPATH = "/home/olab/shakedy/data/ru_wiki_lines.txt"

def main():
    segmenter = TextSharding.NLTKSegmenter()
    with open(WIKI_ONE_ARTICLE_PER_LINE_FILEPATH, "r") as corpus_file:
        corpus_articles = corpus_file.readlines()
    
    new_corpus = open(WIKI_SHORT_LINES_OUTPATH, "w")
    for article in corpus_articles:
        if len(article) < 2: # some aritcles are empty, or just "\n"
            continue

        sentences = segmenter.segment_string(article)
        new_corpus.write('\n'.join(sentences))

if __name__ == "__main__":
    main()
