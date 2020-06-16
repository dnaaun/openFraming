import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers import Trainer, TrainingArguments, InputFeatures

from lda import EXCEL_EXTENSIONS, CSV_EXTENSIONS, TSV_EXTENSIONS


class ClassificationDataset(Dataset):
    """
    Inherits from Torch dataset. Loads data and holds tokenized data for use by a BERT model.
    """
    def __init__(
        self, 
        labels: list, 
        tokenizer: AutoTokenizer, 
        dset_filename:str, 
        content_column: str, 
        label_column: str
    ):
        """
        labels: list of valid labels (can be strings/ints)
        tokenizer: AutoTokenizer object that can tokenize input text
        dset_filename: name of the filename (full filepath) of the dataset being loaded
        content_column: column name of the content to be read
        label_column: column name where the labels can be found
        """
        suffix = dset_filename.split('.')[-1]
        if suffix in EXCEL_EXTENSIONS:
            doc_reader = pd.read_excel
        elif suffix in CSV_EXTENSIONS:
            doc_reader = pd.read_csv
        elif suffix in TSV_EXTENSIONS:
            doc_reader = lambda b: pd.read_csv(b, sep='\t')
        else:
            raise ValueError('File types in directory {} are inconsistent or invalid!'.format(dir_name))

        self.labels = labels
        self.tokenizer = tokenizer
        self.label_map = {label: i for i, label in enumerate(self.labels)}
        df = doc_reader(dset_filename)
        self.encoded_content = self.tokenizer.batch_encode_plus(
        df[content_column], max_length=None, pad_to_max_length=True,
        )
        self.encoded_labels = [self.label_map[label] for label in df[label_column]]
        self.features = []
        for i in range(len(self.encoded_content)):
            inputs = {k: self.encoded_content[k][i] for k in self.encoded_content}
            feature = InputFeatures(**inputs, label=self.encoded_labels[i])
            self.features.append(feature)

    def __len__(self):
        return len(self.encoded_content)

    def __getitem__(self, i: int) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.labels


class ClassifierModel(object):
    """
    Trainable BERT-based classifier given a training & eval set.
    """
    def __init__(self, labels: list, model_path: str, data_dir: str, cache_dir: str):
        """
        labels: list of valid labels used in the dataset
        model_path: name of model being used or filepath to where the model is stored
        data_dir: directory where the training & eval sets are stored, as train.* and eval.*
        cache_dir: directory where cache & output are kept.
        """
        self.cache_dir = cache_dir
        self.model_path = model_path
        self.labels = labels
        self.num_labels = len(labels)
        self.data_dir = data_dir
        self.task_name = 'classification'

        self.config = AutoConfig.from_pretrained(
            self.model_path,
            num_labels=self.num_labels,
            finetuning_task=self.task_name,
            cache_dir=self.cache_dir,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            cache_dir=self.cache_dir,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            from_tf=False,
            config=self.config,
            cache_dir=self.cache_dir,
        )

        self.train_dataset = self.make_dataset('train.csv', 'tweet', 'sentiment')
        self.eval_dataset = self.make_dataset('dev.csv', 'tweet', 'sentiment')

    def make_dataset(self, fname: str, content_column: str, label_column: str) -> ClassificationDataset:
        """
        Creates a Torch dataset object from a file using the built-in tokenizer. 

        Inputs:
            fname: name of the file being used
            content_column: column that contains the text we want to analyze
            label_column: column containing the label

        Returns:
            ClassificationDataset object (which is a Torch dataset underneath)
        """
        return ClassificationDataset(self.labels, self.tokenizer, self.data_dir + fname, content_column, label_column)

    def train(self):
        """
        Train a BERT-based model, using the training set to train & the eval set as validation.
        """
        def simple_accuracy(preds: list, labels: list):
            """
            Checks how often preds == labels
            """
            return (preds == labels).mean()

        def build_compute_metrics_fn() -> Callable[[EvalPrediction], Dict]:
            """
            Get a metrics computation function
            """
            def compute_metrics_fn(p: EvalPrediction):
                """
                Compute accuracy of predictions vs labels
                """
                preds = np.argmax(p.predictions, axis=1)
                return {"acc": simple_accuracy(preds, labels)}

            return compute_metrics_fn

        self.trainer = Trainer(
            model=self.model,
            args=TrainingArguments(do_train=True, do_eval=True, evaluate_during_training=True, output_dir='./output/'),
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=build_compute_metrics_fn(),
        )
        self.trainer.train(
            model_path=self.model_path
        )
        self.trainer.save_model()

    def eval_model(self):
        """
        Wrapper on the trainer.evaluate method; evaluate model's performance on eval set provided by the user.
        """
        return self.trainer.evaluate(eval_dataset=self.eval_dataset)

