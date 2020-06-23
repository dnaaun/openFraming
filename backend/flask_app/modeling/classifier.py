"""Classifier related backend functionality."""
import typing as T

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.utils.data.dataset import Dataset
from transformers import AutoConfig  # type: ignore
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import EvalPrediction
from transformers import InputFeatures
from transformers import Trainer
from transformers import TrainingArguments

from flask_app.modeling.lda import CSV_EXTENSIONS
from flask_app.modeling.lda import EXCEL_EXTENSIONS
from flask_app.modeling.lda import TSV_EXTENSIONS


class ClassificationDataset(Dataset):  # type: ignore
    """Inherits from Torch dataset. Loads and holds tokenized data for a BERT model."""

    def __init__(
        self,
        labels: T.List[str],
        tokenizer: AutoTokenizer,
        label_map: T.Dict[str, int],
        dset_filename: str,
        content_column: str,
        label_column: str,
    ):
        """.

        labels: list of valid labels (can be strings/ints)
        tokenizer: AutoTokenizer object that can tokenize input text
        label_map: maps labels to ints for machine-readability
        dset_filename: name of the filename (full filepath) of the dataset being loaded
        content_column: column name of the content to be read
        label_column: column name where the labels can be found
        """
        suffix = dset_filename.split(".")[-1]
        if suffix in EXCEL_EXTENSIONS:
            doc_reader = pd.read_excel
        elif suffix in CSV_EXTENSIONS:
            doc_reader = pd.read_csv
        elif suffix in TSV_EXTENSIONS:
            doc_reader = lambda b: pd.read_csv(b, sep="\t")
        else:
            raise ValueError(
                f"The file {dset_filename} doesn't have a recognized extension."
            )

        self.labels = labels
        self.label_map = label_map
        self.tokenizer = tokenizer
        df = doc_reader(dset_filename)
        self.len_dset = len(df)

        self.encoded_content = self.tokenizer.batch_encode_plus(
            df[content_column], max_length=None, pad_to_max_length=True,
        )
        if label_column is not None:
            self.encoded_labels = [self.label_map[label] for label in df[label_column]]
        else:
            self.encoded_labels = None
        self.features = []
        for i in range(len(self.encoded_content["token_type_ids"])):
            inputs = {
                k: self.encoded_content[k][i] for k in self.encoded_content.keys()
            }
            if self.encoded_labels is not None:
                feature = InputFeatures(**inputs, label=self.encoded_labels[i])
            else:
                feature = InputFeatures(**inputs, label=None)
            self.features.append(feature)

    def __len__(self) -> int:
        return self.len_dset

    def __getitem__(self, i: int) -> InputFeatures:
        return self.features[i]

    def get_labels(self) -> T.List[str]:
        return self.labels


class ClassifierModel(object):
    """Trainable BERT-based classifier given a training & eval set."""

    def __init__(
        self, labels: T.List[str], model_path: str, data_dir: str, cache_dir: str,
    ):
        """.

        labels: list of valid labels used in the dataset
        model_path: name of model being used or filepath to where the model is stored
        model_path_tokenizer: name or path of tokenizer being used.
        data_dir: directory where the training & eval sets are stored, as train.* and
            eval.*
        cache_dir: directory where cache & output are kept.
        """
        self.cache_dir = cache_dir
        self.model_path = model_path
        self.data_dir = data_dir

        self.labels = labels
        self.num_labels = len(labels)
        self.task_name = "classification"

        self.label_map = {label: i for i, label in enumerate(self.labels)}
        self.label_map_reverse = {i: label for i, label in enumerate(self.labels)}

        self.config = AutoConfig.from_pretrained(
            self.model_path,
            num_labels=self.num_labels,
            finetuning_task=self.task_name,
            cache_dir=self.cache_dir,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, cache_dir=self.cache_dir,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            from_tf=False,
            config=self.config,
            cache_dir=self.cache_dir,
        )

        self.train_dataset = self.make_dataset(
            self.data_dir + "/train.csv", "tweet", "sentiment"
        )
        self.eval_dataset = self.make_dataset(
            self.data_dir + "/dev.csv", "tweet", "sentiment"
        )

    def make_dataset(
        self, fname: str, content_column: str, label_column: str
    ) -> ClassificationDataset:
        """Create a Torch dataset object from a file using the built-in tokenizer.

        Inputs:
            fname: name of the file being used
            content_column: column that contains the text we want to analyze
            label_column: column containing the label

        Returns:
            ClassificationDataset object (which is a Torch dataset underneath)
        """
        return ClassificationDataset(
            self.labels,
            self.tokenizer,
            self.label_map,
            fname,
            content_column,
            label_column,
        )

    def train(self) -> None:
        """Train a BERT-based model, using the training set to train & the eval set as
        validation.
        """

        def simple_accuracy(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
            """
            Checks how often preds == labels
            """
            return (preds == labels).mean()

        def build_compute_metrics_fn() -> T.Callable[
            [EvalPrediction], T.Dict[str, np.ndarray]
        ]:
            """
            Get a metrics computation function
            """

            def compute_metrics_fn(p: EvalPrediction) -> T.Dict[str, np.ndarray]:
                """
                Compute accuracy of predictions vs labels
                """
                preds = np.argmax(p.predictions, axis=1)
                labels = p.label_ids
                return {"acc": simple_accuracy(preds, labels)}

            return compute_metrics_fn

        self.trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                do_train=True,
                do_eval=True,
                evaluate_during_training=True,
                output_dir="./output/",
            ),
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=build_compute_metrics_fn(),
        )
        self.trainer.train(model_path=self.model_path)
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.trainer.args.output_dir)

    def eval_model(self) -> T.Dict[str, float]:
        """
        Wrapper on the trainer.evaluate method; evaluate model's performance on eval set
        provided by the user.
        """
        return self.trainer.evaluate(eval_dataset=self.eval_dataset)  # type: ignore

    def _run_inference(self, inference_dset: ClassificationDataset) -> T.List[str]:
        """
        Given a dataset, return the labels 
        (as drawn from the original set of labels given by the user) 
        that the model predicts.

        Inputs:
            inference_dset: ClassificationDataset object. May or may not have labels.
        Outputs: 
            list of predicted labels (congruent with user input).
        """
        preds = None
        with torch.no_grad():
            for inputs in inference_dset:
                inputs_new = {}
                inputs_new["input_ids"] = torch.Tensor([inputs.input_ids]).long()
                inputs_new["token_type_ids"] = torch.Tensor(
                    [inputs.token_type_ids]
                ).long()
                inputs_new["attention_mask"] = torch.Tensor([inputs.attention_mask])
                outputs = self.model(**inputs_new)
                logits = outputs[0]
                if preds is None:
                    preds = torch.nn.functional.softmax(logits).detach()
                else:
                    preds = torch.cat((preds, logits.detach()), dim=0)

        preds = preds.cpu().numpy()
        preds = [self.label_map_reverse[np.argmax(i)] for i in preds]
        return preds

    def run_inference_no_eval(
        self, inference_dset_path: str, text_col: str
    ) -> T.List[str]:
        """
        Given a path to a dataset and the column containing text, 
        provide the labels predicted by the model.

        Inputs:
            inference_dset_path: absolute filepath of inference dataset
            text_col: column containing the text we'll analyze

        Outputs:
            list of predictions (as user-supplied labels)
        """
        inference_dset = self.make_dataset(inference_dset_path, text_col, None)
        return self._run_inference(inference_dset)

    def run_inference_and_evaluate(
        self, inference_dset_path: str, text_col: str, label_col: str
    ) -> T.Dict[str, float]:
        """
        Given a path to a dataset and the columns containing text & labels,
        provide the labels predicted by the model and some basic metrics.

        Inputs:
            inference_dset_path: absolute path to inference dataset
            text_col: column containing the text we'll analyze
            label_col: column containing the labels

        Outputs:
            list of predictions (as user-supplied labels)
            a dict containing basic classification metrics
        """
        inference_dset = self.make_dataset(inference_dset_path, text_col, label_col)
        labels = inference_dset.encoded_labels
        preds = self._run_inference(inference_dset)
        return preds, classification_report(labels, preds, output_dict=True)
