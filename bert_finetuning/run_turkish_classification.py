# coding=utf-8
# Copyright 2021 Intel Corporation. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import json
import logging
import os
import random
import sys
from argparse import Namespace
from pretraining.args.deepspeed_args import remove_cuda_compatibility_for_kernel_compilation
from pretraining.modeling import BertForSequenceClassification
from pretraining.configs import PretrainedBertConfig
from dataclasses import dataclass, field
from typing import Optional
import uuid

import numpy as np
import transformers
from torch.utils.data import DataLoader
from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import SchedulerType, is_main_process

task_to_keys = {
    "tr_reviews" : ("sentence", "sentiment"),
}

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )

@dataclass
class FinetuneTrainingArguments(TrainingArguments):
    group_name: Optional[str] = field(default=None, metadata={"help": "W&B group name"})
    project_name: Optional[str] = field(default=None, metadata={"help": "Project name (W&B)"})
    early_stopping_patience: Optional[int] = field(
        default=-1, metadata={"help": "Early stopping patience value (default=-1 (disable))"}
    )
    # overriding to be True, for consistency with final_eval_{metric_name}
    fp16_full_eval: bool = field(
        default=True,
        metadata={"help": "Whether to use full 16-bit precision evaluation instead of 32-bit"},
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    unique_run_id = str(uuid.uuid1())

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, FinetuneTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                            "Use --overwrite_output_dir to overcome.")

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                            datefmt="%m/%d/%Y %H:%M:%S",
                            level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,)

    # Log on each process the small summary:
    logger.warning(f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
                    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}")
    
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: We have single sentence classification, only one single column is not label.
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    logger.info("Splitting raw_dataset to {} for training, {} for eval, {} for pred".format(80, 10, 10))
    raw_training = load_dataset("turkish_product_reviews", 
                                        split='train')

    training_dataset = load_dataset("turkish_product_reviews", 
                                        split='train[:80%]')

    eval_dataset = load_dataset("turkish_product_reviews", 
                                        split='train[-20%:-10%]')

    predict_dataset = load_dataset("turkish_product_reviews", 
                                        split='train[-10%:]')

    num_labels = 2
    label_list = [0, 1]
    sentence_key = "sentence"
    label_key = "sentiment"

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    pretrain_run_args = json.load(open(f"{model_args.model_name_or_path}/args.json", "r"))

    def get_correct_ds_args(pretrain_run_args):
        ds_args = Namespace()

        for k, v in pretrain_run_args.items():
            setattr(ds_args, k, v)

        # to enable HF integration
        #         ds_args.huggingface = True
        return ds_args

    ds_args = get_correct_ds_args(pretrain_run_args)

    # in so, deepspeed is required
    if ("deepspeed_transformer_kernel" in pretrain_run_args and pretrain_run_args["deepspeed_transformer_kernel"]):
        logger.warning("Using deepspeed_config due to kernel usage")
        remove_cuda_compatibility_for_kernel_compilation()

    if os.path.isdir(model_args.model_name_or_path):
        config = PretrainedBertConfig.from_pretrained(model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                                                        num_labels=num_labels,
                                                        finetuning_task=data_args.task_name,
                                                        cache_dir=model_args.cache_dir,)
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name
                                                    if model_args.tokenizer_name
                                                    else model_args.model_name_or_path,
                                                    cache_dir=model_args.cache_dir,)
        model = BertForSequenceClassification.from_pretrained(model_args.model_name_or_path,
                                                                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                                                                config=config,
                                                                cache_dir=model_args.cache_dir,
                                                                args=ds_args,)
    else:
        config = AutoConfig.from_pretrained(model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                                            num_labels=num_labels,
                                            finetuning_task=data_args.task_name,
                                            cache_dir=model_args.cache_dir,
                                            revision=model_args.model_revision,
                                            use_auth_token=True if model_args.use_auth_token else None,)
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name
                                                    if model_args.tokenizer_name
                                                    else model_args.model_name_or_path,
                                                    cache_dir=model_args.cache_dir,
                                                    use_fast=model_args.use_fast_tokenizer,
                                                    revision=model_args.model_revision,
                                                    use_auth_token=True if model_args.use_auth_token else None,)
        model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path,
                                                                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                                                                    config=config,
                                                                    cache_dir=model_args.cache_dir,
                                                                    revision=model_args.model_revision,
                                                                    use_auth_token=True if model_args.use_auth_token else None,)

    # Preprocessing the datasets
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
        max_length = data_args.max_seq_length
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False
        max_length = None

    def preprocess_function(examples):
        # Tokenize the texts
        tokenized_inputs = tokenizer(
            examples[sentence_key],
            padding=padding,
            truncation=True,
            max_length=max_length)
        return tokenized_inputs

    # we need the dataset contain "label", not "sentiment"
    training_dataset = training_dataset.map(preprocess_function, 
                                            batched=True, 
                                            load_from_cache_file=not data_args.overwrite_cache)
    # SY: Cut data to have them in equal size
    positive_raw = raw_training.filter(lambda example: example['sentiment'] == 1)
    negative_raw = raw_training.filter(lambda example: example['sentiment'] == 0)
    logger.error("all -- positive: {}, negative:{}\n".format(len(positive_raw), len(negative_raw)))
    
    positive_training = training_dataset.filter(lambda example: example['sentiment'] == 1)
    negative_training = training_dataset.filter(lambda example: example['sentiment'] == 0)
    logger.error("training -- positive: {}, negative:{}\n".format(len(positive_training), len(negative_training)))
    
    positive_eval = eval_dataset.filter(lambda example: example['sentiment'] == 1)
    negative_eval = eval_dataset.filter(lambda example: example['sentiment'] == 0)
    logger.error("eval -- positive: {}, negative:{}\n".format(len(positive_eval), len(negative_eval)))

    positive_predict = predict_dataset.filter(lambda example: example['sentiment'] == 1)
    negative_predict = predict_dataset.filter(lambda example: example['sentiment'] == 0)
    logger.error("positive: {}, negative:{}\n".format(len(positive_predict), len(negative_predict)))
    exit(-1)

    training_dataset = training_dataset.rename_column("sentiment", "label")

    eval_dataset = eval_dataset.map(preprocess_function, 
                                            batched=True, 
                                            load_from_cache_file=not data_args.overwrite_cache)
    eval_dataset = eval_dataset.rename_column("sentiment", "label")

    predict_dataset = predict_dataset.map(preprocess_function, 
                                            batched=True, 
                                            load_from_cache_file=not data_args.overwrite_cache)
    predict_dataset = predict_dataset.rename_column("sentiment", "label")

    # Log a few random samples from the training set:
    for index in random.sample(range(len(training_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {training_dataset[index]}.")

    # Get the metric function
    metric_acc = load_metric("accuracy")
    metric_f1 = load_metric("f1")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1) #np.squeeze(preds) #if is_regression else 
        
        result_acc = metric_acc.compute(predictions=preds, references=p.label_ids)
        logger.info("acc: {}".format(result_acc))
        #if len(result_acc) > 1:
        #        result_acc["combined_score"] = np.mean(list(result_acc.values())).item()

        result_f1 = metric_f1.compute(predictions=preds, references=p.label_ids)
        logger.info("f1: {}".format(result_f1))
        #if len(result_f1) > 1:
        #        result_f1["combined_score"] = np.mean(list(result_f1.values())).item()
        
        result = result_acc
        result.update(result_f1)
        return result
        #else:
        #    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # init early stopping callback and metric to monitor
    callbacks = None
    if training_args.early_stopping_patience > 0:
        early_cb = EarlyStoppingCallback(training_args.early_stopping_patience)
        callbacks = [early_cb]

        metric_to_monitor = "accuracy"
        setattr(training_args, "metric_for_best_model", metric_to_monitor)
        logger.info("Initialized early stopping callback.")
    else:
        logger.info("Not need early stopping callback.")

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    logger.info("Data collector is ready.")

    # Initialize our Trainer
    logger.info("training instance: {}".format(training_dataset[1]))
    # How inputs to training step look?
    '''dataloader = DataLoader(training_dataset,
                                    batch_size=8,
                                    collate_fn=data_collator)
    for inputs in dataloader:
        logger.error("inputs: {}".format(inputs))'''

    trainer = Trainer(model=model,
                        args=training_args,
                        train_dataset=training_dataset,
                        eval_dataset=eval_dataset,
                        compute_metrics=compute_metrics,
                        tokenizer=tokenizer,
                        callbacks=callbacks,
                        data_collator=data_collator,)

    # Training
    if training_args.do_train:
        trainer.train()

    # Evaluation
    if training_args.do_eval:
        print("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        logger.info("Saving expected predictions...")
        output_expected_predictions_file = os.path.join(training_args.output_dir, "expected_predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_expected_predictions_file, "w") as writer:
                for instance in predict_dataset:
                    writer.write("sentence: {}, prediction: {}\n".format(instance["sentence"], instance["label"]))

        predictions, _, metrics= trainer.predict(predict_dataset)
        predictions = np.argmax(predictions, axis=1)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # Save predictions
        output_predictions_file = os.path.join(training_args.output_dir, "predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_predictions_file, "w") as writer:
                for prediction in predictions:
                    writer.write("{}".format(prediction) + "\n")

if __name__ == "__main__":
    main()
