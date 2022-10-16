The academic-budget-bert comes only with builtin run_glue.py.
In order to finetune NER, XNLI or QA, you should adapt by yourself the huggingface scripts (can found them here: https://github.com/huggingface/transformers/tree/main/examples/pytorch, some list comes later), which is not a trivial task.

We wanted to finetune NER and XNLI too, so we adapt the original transformers run_xnli.py and run_ner.py, the way academic-budget-bert already adapt the run_glue.py for us.

You can find in this repo the adapted run_ner.py and run_xnli.py.
In addition, you can find here "modeling.py", that is required to replace the "academic-budget-bert/pretraining/modeling.py".

If you ever would like to adapt more tasks, check the following sections out:

# Some transformers useful original links:
------------------------------------------
1. GLUE FT: https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py
2. XNLI FT: https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_xnli.py
3. SWAG FT: https://github.com/huggingface/transformers/blob/main/examples/pytorch/multiple-choice/run_swag.py
4. NER FT:  https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py
5. Extractive QA FT: https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa.py
and there are much more in the git...

# Adaptations:
--------------
You can look at "run_ner.py" or "modeling.py" and search for "SY:", the adaptations I did are commented with explanation.
Then, I recommend you firstly to understand what the script tries to do in high level.

Some examples for changes:
1. Adaptations for the dataset - This is not exactly related to "academic-budget-bert", but not all transformers script works for all datasets. For example, in turkish polyglot_ner there was no validation and test set, so we had to split the train set to 80% train, 10% eval and 10% predict datasets.
There were other scripts where the datasets column names were not true (for example "sentiment" column instead of the expected "label" column name), so go through it and make sure it tailored to your dataset.
2. Some scripts require additional files, for example "run_qa.py" requires you bring with you "utils_qa.py" too (it also found in the transformers script), so make sure you understand your script's requirements.
3. In modeling.py, you should put the BERT finetuning model (BertForTokenClassification for NER for example).
I copied the right ones from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py.
