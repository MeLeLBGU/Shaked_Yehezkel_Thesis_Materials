# SaGe:
-------
Most current popular subword tokenizers are trained based on word frequency statistics over a corpus, without considering information about co-occurrence or context. Nevertheless, the resulting vocabularies are used in language models' highly contextualized settings. We present SaGe, a tokenizer that tailors subwords for their downstream use by baking in the contextualized signal at the vocabulary creation phase. We show that SaGe does a better job than current widespread tokenizers in keeping token contexts cohesive, while not incurring a large price in terms of encoding efficiency or domain robustness. SaGe improves performance on English GLUE classification tasks as well as on NER, and on Inference and NER in Turkish, demonstrating its robustness to language properties such as morphological exponence and agglutination.
(https://arxiv.org/abs/2210.07095)

## Vocabulary Creation:
----------------------- 
The code that implements SaGe algorithm.

## BERT Pretraining:
--------------------
Explains how to pretrain bert model using the created vocabulary, and using the academic-budget-bert repository.

## BERT Finetuning:
-------------------
Explains how to finetune english for GLUE tasks using academic-budget-bert code, and how to adapt other huugingface transformers scripts in order to finetune other languages and tasks.

## Analysis:
------------
Some manual analysis scripts we had to plot graphs and histograms.
