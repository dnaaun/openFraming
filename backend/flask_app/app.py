"""All the flask api endpoints."""
import csv
import io
import logging
import os
import sys
import typing as T
from collections import Counter
from pathlib import Path

import pandas as pd  # type: ignore
import typing_extensions as TT
from flask import current_app
from flask import Flask
from flask_restful import Api  # type: ignore
from flask_restful import reqparse
from flask_restful import Resource
from playhouse.flask_utils import get_object_or_404
from sklearn import model_selection  # type: ignore
from typing_extensions import TypedDict
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import BadRequest
from werkzeug.exceptions import HTTPException
from werkzeug.exceptions import NotFound

from flask_app import db
from flask_app import utils
from flask_app.modeling.train_queue import Scheduler

API_URL_PREFIX = "/api"

logger = logging.getLogger(__name__)


class UnprocessableEntity(HTTPException):
    """."""

    code = 422
    description = "The entity supplied has errors and cannot be processed."


class AlreadyExists(HTTPException):
    """."""

    code = 403
    description = "The resource already exists."


class BaseResource(Resource):
    """Every resource derives from this.

    Attributes:
        url:
    """

    url: str

    @staticmethod
    def _write_headers_and_data_to_csv(
        headers: T.List[str], data: T.List[T.List[str]], csvfile: Path
    ) -> None:

        with csvfile.open("w") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(data)

    @staticmethod
    def _validate_serializable_list_value(val: T.Any) -> str:
        if not isinstance(val, str):
            raise ValueError("must be str")
        if "," in val:
            raise ValueError("can't contain commas.")
        return val


class ClassifierStatusJson(TypedDict):
    classifier_id: int
    classifier_name: str
    category_names: T.List[str]
    trained_by_openFraming: bool
    status: TT.Literal["not_begun", "training", "completed"]
    metrics: T.Optional[utils.ClassifierMetrics]
    # TODO: Update backend README to reflect API change for line above.


class ClassifierRelatedResource(BaseResource):
    """Base class to define utility functions related to classifiers."""

    @staticmethod
    def _classifier_status(clsf: db.Classifier) -> ClassifierStatusJson:
        """Process a Classifier instance and format it into the API spec."""
        metrics: T.Optional[utils.ClassifierMetrics] = None
        status: TT.Literal["not_begun", "completed", "training"] = "not_begun"
        if clsf.train_set is not None:
            assert clsf.dev_set is not None
            if clsf.train_set.training_or_inference_completed:
                assert clsf.dev_set.training_or_inference_completed
                assert clsf.dev_set.metrics is not None
                status = "completed"
                metrics = utils.ClassifierMetrics(
                    accuracy=clsf.dev_set.metrics.accuracy,
                    macro_f1_score=clsf.dev_set.metrics.macro_f1_score,
                    macro_precision=clsf.dev_set.metrics.macro_precision,
                    macro_recall=clsf.dev_set.metrics.macro_recall,
                )
            else:
                status = "training"

        category_names = clsf.category_names

        return ClassifierStatusJson(
            {
                "classifier_id": clsf.classifier_id,
                "classifier_name": clsf.name,
                "trained_by_openFraming": clsf.trained_by_openFraming,
                "category_names": category_names,
                "status": status,
                "metrics": metrics,
            }
        )


class OneClassifier(ClassifierRelatedResource):

    url = "/classifiers/<int:classifier_id>"

    def get(self, classifier_id: int) -> ClassifierStatusJson:
        clsf = get_object_or_404(
            db.Classifier, db.Classifier.classifier_id == classifier_id
        )
        return self._classifier_status(clsf)


class Classifiers(ClassifierRelatedResource):
    """Create a classifer, get a list of classifiers."""

    url = "/classifiers/"

    def __init__(self) -> None:
        """Set up request parser."""
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument(name="name", type=str, required=True)

        self.reqparse.add_argument(
            name="category_names",
            type=self._validate_serializable_list_value,
            action="append",
            required=True,
            help="",
        )

    def post(self) -> ClassifierStatusJson:
        """Create a classifier.

        req_body:
            json:
                {
                    "name": str,
                    "category_names": T.List[str]
                }

        """
        args = self.reqparse.parse_args()
        if (
            len(args["category_names"]) < 2
        ):  # I don't know how to do this validation in the
            # RequestParser
            raise BadRequest("must be at least two categories.")

        category_names = args["category_names"]
        name = args["name"]
        clsf = db.Classifier.create(name=name, category_names=category_names)
        clsf.save()
        utils.Files.classifier_dir(classifier_id=clsf.classifier_id, ensure_exists=True)
        return self._classifier_status(clsf)

    def get(self) -> T.List[ClassifierStatusJson]:
        """Get a list of classifiers."""
        res = [self._classifier_status(clsf) for clsf in db.Classifier.select()]
        return res


class ClassifiersTrainingFile(ClassifierRelatedResource):
    """Upload training data to the classifier."""

    url = "/classifiers/<int:classifier_id>/training/file"

    def __init__(self) -> None:
        """Set up request parser."""
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument(
            name="file", type=FileStorage, required=True, location="files"
        )

    def post(self, classifier_id: int) -> ClassifierStatusJson:
        """Upload a training set for classifier, and start training.

        Body:
            FormData: with "file" item. 

        Raises:
            BadRequest
            UnprocessableEntity

        """
        args = self.reqparse.parse_args()
        file_: FileStorage = args["file"]

        try:
            classifier = db.Classifier.get(db.Classifier.classifier_id == classifier_id)
        except db.Classifier.DoesNotExist:
            raise NotFound("classifier not found.")

        if classifier.train_set is not None:
            raise AlreadyExists("This classifier already has a training set.")

        table_headers, table_data = self._validate_training_file_and_get_data(
            classifier.category_names, file_
        )
        file_.close()
        # Split into train and dev
        ss = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.2)
        X, y = zip(*table_data)
        train_indices, dev_indices = next(ss.split(X, y))

        train_data = [table_data[i] for i in train_indices]
        dev_data = [table_data[i] for i in dev_indices]

        train_file = utils.Files.classifier_train_set_file(classifier_id)
        self._write_headers_and_data_to_csv(table_headers, train_data, train_file)
        dev_file = utils.Files.classifier_dev_set_file(classifier_id)
        self._write_headers_and_data_to_csv(table_headers, dev_data, dev_file)

        classifier.train_set = db.LabeledSet()
        classifier.dev_set = db.LabeledSet()
        classifier.train_set.save()
        classifier.dev_set.save()
        classifier.save()

        # Refresh classifier
        classifier = db.Classifier.get(db.Classifier.classifier_id == classifier_id)

        model_scheduler: Scheduler = current_app.config["SCHEDULER"]

        # TODO: Add a check to make sure model training didn't start already and crashed

        model_scheduler.add_classifier_training(
            classifier_id=classifier.classifier_id,
            labels=classifier.category_names,
            model_path=utils.TRANSFORMERS_MODEL,
            train_file=str(utils.Files.classifier_train_set_file(classifier_id)),
            dev_file=str(utils.Files.classifier_dev_set_file(classifier_id)),
            cache_dir=str(current_app.config["TRANSFORMERS_CACHE_DIR"]),
            output_dir=str(
                utils.Files.classifier_output_dir(classifier_id, ensure_exists=True)
            ),
        )

        return self._classifier_status(classifier)

    @staticmethod
    def _validate_training_file_and_get_data(
        category_names: T.List[str], file_: FileStorage
    ) -> T.Tuple[T.List[str], T.List[T.List[str]]]:
        """Validate user uploaded file and return uploaded validated data.

        Args:
            category_names: The categories for the classifier.
            file_: uploaded file.

        Returns:
            table_headers: A list of length 2.
            table_data: A list of lists of length 2.
        """
        # TODO: Write tests for all of these!

        table = utils.Validate.csv_and_get_table(T.cast(io.BytesIO, file_))

        utils.Validate.table_has_no_empty_cells(table)
        utils.Validate.table_has_num_columns(table, 2)
        utils.Validate.table_has_headers(table, [utils.CONTENT_COL, utils.LABEL_COL])

        table_headers, table_data = table[0], table[1:]

        min_num_examples = int(len(table_data) * utils.TEST_SET_SPLIT)
        if len(table_data) < min_num_examples:
            raise BadRequest(
                f"We need at least {min_num_examples} labelled examples for this issue."
            )

        # TODO: Low priority: Make this more efficient.
        category_names_counter = Counter(category for _, category in table_data)

        unique_category_names = category_names_counter.keys()
        if set(category_names) != unique_category_names:
            # TODO: Lower case category names before checking.
            # TODO: More helpful error messages when there is an error with the
            # the categories in an uploaded training file.
            raise UnprocessableEntity(
                "The categories for this classifier are"
                f" {category_names}. But the uploaded file either"
                " has some categories missing, or has categories in addition to the"
                " ones indicated."
            )

        categories_with_less_than_two_exs = [
            category for category, count in category_names_counter.items() if count < 2
        ]
        if categories_with_less_than_two_exs:
            raise UnprocessableEntity(
                "There are less than two examples with the categories: "
                f"{','.join(categories_with_less_than_two_exs)}."
                " We need at least two examples per category."
            )

        return table_headers, table_data


class ClassifierTestSetStatusJson(TypedDict):
    classifier_id: int
    test_set_id: int
    test_set_name: str
    status: TT.Literal["not_begun", "predicting", "completed"]


class ClassifierTestSetRelatedResource(ClassifierRelatedResource):
    @staticmethod
    def _test_set_status(test_set: db.TestSet) -> ClassifierTestSetStatusJson:
        status: TT.Literal["not_begun", "predicting", "completed"] = "not_begun"
        if test_set.inference_began:
            if test_set.inference_completed:
                status = "completed"
            else:
                status = "predicting"

        return ClassifierTestSetStatusJson(
            classifier_id=test_set.classifier.classifier_id,
            test_set_id=test_set.id_,
            test_set_name=test_set.name,
            status=status,
        )


class OneClassifierTestSet(ClassifierTestSetRelatedResource):

    url = "/classifiers/<int:classifier_id>/test_sets/<int:test_set_id>"

    def get(self, classifier_id: int, test_set_id: int) -> ClassifierTestSetStatusJson:
        test_set = get_object_or_404(db.TestSet, db.TestSet.id_ == test_set_id)
        if test_set.classifier.classifier_id != classifier_id:
            raise NotFound("!test set not found.")
        return self._test_set_status(test_set)


class ClassifiersTestSets(ClassifierTestSetRelatedResource):
    """Upload training data to the classifier."""

    url = "/classifiers/<int:classifier_id>/test_sets/"

    def __init__(self) -> None:
        """Set up request parser."""
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument(name="test_set_name", type=str, required=True)

    def get(self, classifier_id: int) -> T.List[ClassifierTestSetStatusJson]:
        clsf = get_object_or_404(
            db.Classifier, db.Classifier.classifier_id == classifier_id
        )

        return [self._test_set_status(test_set) for test_set in clsf.test_sets]

    def post(self, classifier_id: int) -> ClassifierTestSetStatusJson:
        """Create a test set

        Body:
            {
                "test_set_name": str,
            }

        Raises:
            BadRequest
            NotFound
        """
        args = self.reqparse.parse_args()
        test_set_name: str = args["test_set_name"]

        try:
            classifier = db.Classifier.get(db.Classifier.classifier_id == classifier_id)
        except db.Classifier.DoesNotExist:
            raise NotFound("classifier not found.")

        if classifier.train_set is None:
            assert classifier.dev_set is None
            raise BadRequest("This classifier has not been trained yet.")
        elif not classifier.train_set.training_or_inference_completed:
            assert classifier.dev_set is not None
            assert not classifier.dev_set.training_or_inference_completed
            raise BadRequest("This classifier's training has not been completed yet.")

        test_set = db.TestSet(classifier=classifier, name=test_set_name)
        test_set.save()

        # Create directory for test set
        utils.Files.classifier_test_set_dir(
            classifier_id, test_set.id_, ensure_exists=True
        )

        return self._test_set_status(test_set)


class ClassifiersTestSetsFile(ClassifierTestSetRelatedResource):
    """Upload training data to the classifier."""

    url = "/classifiers/<int:classifier_id>/test_sets/<int:test_set_id>/file"

    def __init__(self) -> None:
        """Set up request parser."""
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument(
            name="file", type=FileStorage, required=True, location="files"
        )

    def post(self, classifier_id: int, test_set_id: int) -> ClassifierTestSetStatusJson:
        """Upload a training set for classifier, and start training.

        Body:
            FormData: with "file" item. 

        Raises:
            BadRequest
            UnprocessableEntity
            NotFound
        """
        args = self.reqparse.parse_args()
        file_: FileStorage = args["file"]

        test_set = get_object_or_404(db.TestSet, db.TestSet.id_ == test_set_id)

        if test_set.classifier.classifier_id != classifier_id:
            raise NotFound("The test set id was not found.")

        if test_set.inference_began:
            raise AlreadyExists("The file for this test set has already been uploaded.")

        table_headers, table_data = self._validate_test_file_and_get_data(file_)

        test_file = utils.Files.classifier_test_set_file(classifier_id, test_set_id)
        self._write_headers_and_data_to_csv(table_headers, table_data, test_file)

        test_set.inference_began = True
        test_set.save()

        model_scheduler: Scheduler = current_app.config["SCHEDULER"]

        # TODO: Add a check to make sure model training didn't start already and crashed

        test_output_file = utils.Files.classifier_test_set_predictions_file(
            classifier_id, test_set_id
        )
        model_path = utils.Files.classifier_output_dir(classifier_id)

        model_scheduler.add_classifier_prediction(
            test_set_id=test_set_id,
            labels=test_set.classifier.category_names,
            model_path=str(model_path),
            test_file=str(test_file),
            cache_dir=str(current_app.config["TRANSFORMERS_CACHE_DIR"]),
            test_output_file=str(test_output_file),
        )

        return self._test_set_status(test_set)

    @staticmethod
    def _validate_test_file_and_get_data(
        file_: FileStorage,
    ) -> T.Tuple[T.List[str], T.List[T.List[str]]]:
        """Validate user uploaded file and return validated data.

        Args:
            file_: uploaded file.
            category_names: The categories for the classifier.

        Returns:
            table_headers: A list of length 2.
            table_data: A list of lists of length 2.
        """

        table = utils.Validate.csv_and_get_table(T.cast(io.BytesIO, file_))

        utils.Validate.table_has_no_empty_cells(table)
        utils.Validate.table_has_num_columns(table, 1)
        utils.Validate.table_has_headers(table, [utils.CONTENT_COL])
        table_headers, table_data = table[0], table[1:]

        min_num_examples = 1
        if len(table_data) < min_num_examples:
            raise BadRequest(
                f"We need at least {min_num_examples} examples to run prediction on."
            )

        return table_headers, table_data


class TopicModelStatusJson(TypedDict):
    topic_model_id: int
    topic_model_name: str
    num_topics: int
    topic_names: T.Optional[T.List[str]]
    status: TT.Literal["not_begun", "training", "topics_to_be_named", "completed"]
    # TODO: Update backend README to reflect API change for line above.


class OneTopicPreviewJson(TypedDict):
    keywords: T.List[str]
    examples: T.List[str]


class TopicModelPreviewJson(TopicModelStatusJson):
    topic_previews: T.List[OneTopicPreviewJson]


class TopicModelRelatedResource(BaseResource):
    """Base class to define utility functions related to classifiers."""

    @staticmethod
    def _topic_model_status_json(topic_mdl: db.TopicModel) -> TopicModelStatusJson:
        topic_names = topic_mdl.topic_names
        status: TT.Literal["not_begun", "training", "topics_to_be_named", "completed"]
        if topic_mdl.lda_set is None:
            status = "not_begun"
        else:
            if topic_mdl.lda_set.lda_completed:

                if topic_names is None:
                    status = "topics_to_be_named"
                else:
                    status = "completed"
                status = "completed"
            else:
                status = "training"

        return TopicModelStatusJson(
            topic_model_name=topic_mdl.name,
            topic_model_id=topic_mdl.id_,
            num_topics=topic_mdl.num_topics,
            topic_names=topic_names,
            status=status,
        )

    @staticmethod
    def _validate_topic_model_finished_training(topic_mdl: db.TopicModel) -> None:
        if topic_mdl.lda_set is None:
            raise BadRequest("Topic model has not started training yet.")
        elif not topic_mdl.lda_set.lda_completed:
            raise BadRequest("Topic model has not finished trianing yet.")


class OneTopicModel(TopicModelRelatedResource):

    url = "/topic_models/<int:topic_model_id>"

    def get(self, topic_model_id: int) -> TopicModelStatusJson:
        topic_mdl = get_object_or_404(
            db.TopicModel, db.TopicModel.id_ == topic_model_id
        )
        return self._topic_model_status_json(topic_mdl)


class TopicModels(TopicModelRelatedResource):

    url = "/topic_models/"

    def __init__(self) -> None:
        """Set up request parser."""
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument(name="topic_model_name", type=str, required=True)

        def greater_than_1(x: T.Any) -> int:
            int_x = int(x)
            if int_x < 1:
                raise ValueError("must be greater than 1")
            return int_x

        self.reqparse.add_argument(
            name="num_topics", type=greater_than_1, required=True
        )

    def post(self) -> TopicModelStatusJson:
        """Create a classifier.

        req_body:
            json:
                {
                    "topic_model_name": str,
                    "num_topics": int,
                } 

        """
        args = self.reqparse.parse_args()
        topic_mdl = db.TopicModel.create(
            name=args["topic_model_name"], num_topics=args["num_topics"]
        )
        topic_mdl.save()
        utils.Files.topic_model_dir(id_=topic_mdl.id_, ensure_exists=True)
        return self._topic_model_status_json(topic_mdl)

    def get(self) -> T.List[TopicModelStatusJson]:
        res = [
            self._topic_model_status_json(topic_mdl)
            for topic_mdl in db.TopicModel.select()
        ]
        return res


class TopicModelsTrainingFile(TopicModelRelatedResource):

    url = "/topic_models/<int:id_>/training/file"

    def __init__(self) -> None:
        """Set up request parser."""
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument(
            name="file", type=FileStorage, required=True, location="files"
        )

    def post(self, id_: int) -> TopicModelStatusJson:
        args = self.reqparse.parse_args()
        file_: FileStorage = args["file"]

        try:
            topic_mdl = db.TopicModel.get(db.TopicModel.id_ == id_)
        except db.Classifier.DoesNotExist:
            raise NotFound("classifier not found.")

        if topic_mdl.lda_set is not None:
            raise AlreadyExists("This topic model already has a training set.")

        table_headers, table_data = self._validate_and_get_training_file(file_)
        file_.close()

        train_file = utils.Files.topic_model_training_file(id_)
        self._write_headers_and_data_to_csv(table_headers, table_data, train_file)

        scheduler: Scheduler = current_app.config["SCHEDULER"]

        scheduler.add_topic_model_training(
            topic_model_id=topic_mdl.id_,
            training_file=str(train_file),
            num_topics=topic_mdl.num_topics,
            fname_keywords=str(utils.Files.topic_model_keywords_file(id_)),
            fname_topics_by_doc=str(utils.Files.topic_model_topics_by_doc_file(id_)),
            mallet_bin_directory=str(current_app.config["MALLET_BIN_DIRECTORY"]),
        )
        topic_mdl.lda_set = db.LDASet()
        topic_mdl.lda_set.save()
        topic_mdl.save()

        # Refresh classifier
        topic_mdl = db.TopicModel.get(db.TopicModel.id_ == id_)

        # model_scheduler: ModelScheduler = current_app.config["SCHEDULER"]

        return self._topic_model_status_json(topic_mdl)

    @staticmethod
    def _validate_and_get_training_file(
        file_: FileStorage,
    ) -> T.Tuple[T.List[str], T.List[T.List[str]]]:
        """Validate user input and return uploaded CSV data.

        Args:
            file_: uploaded file.

        Returns:
            table_headers: A list of length 2.
            table_data: A list of lists of length 2.
        """
        # TODO: Write tests for all of these!

        table = utils.Validate.csv_and_get_table(T.cast(io.BytesIO, file_))

        utils.Validate.table_has_num_columns(table, 1)
        utils.Validate.table_has_headers(table, [utils.CONTENT_COL])
        utils.Validate.table_has_no_empty_cells(table)

        table_headers, table_data = table[0], table[1:]
        # add the ID column to the table, necessary because of how the
        # flask_app.modeling.lda.LDAModeler is coded up right now.
        table_headers = [utils.ID_COL] + table_headers
        table_data = [
            [str(row_num)] + row for row_num, row in enumerate(table_data, start=1)
        ]

        if len(table_data) < utils.MINIMUM_LDA_EXAMPLES:
            raise BadRequest(
                f"We need at least {utils.MINIMUM_LDA_EXAMPLES} for a topic model."
            )

        return table_headers, table_data


class TopicModelsTopicsNames(TopicModelRelatedResource):

    url = "/topic_models/<int:id_>/topics/names"

    def __init__(self) -> None:
        self.reqparse = reqparse.RequestParser()

        self.reqparse.add_argument(
            name="topic_names",
            type=self._validate_serializable_list_value,
            action="append",
            required=True,
            help="",
        )

    def post(self, id_: int) -> TopicModelStatusJson:
        args = self.reqparse.parse_args()
        topic_names: T.List[str] = args["topic_names"]
        topic_mdl = get_object_or_404(db.TopicModel, db.TopicModel.id_ == id_)

        self._validate_topic_model_finished_training(topic_mdl)
        if len(topic_names) != topic_mdl.num_topics:
            raise BadRequest(
                f"Topic model has {topic_mdl.num_topics} topics, but {len(topic_names)} topics were provided."
            )

        topic_mdl.topic_names = topic_names
        topic_mdl.save()
        return self._topic_model_status_json(topic_mdl)


class TopicModelsTopicsPreview(TopicModelRelatedResource):

    url = "/topic_models/<int:id_>/topics/preview"

    def get(self, id_: int) -> TopicModelPreviewJson:
        topic_mdl = get_object_or_404(db.TopicModel, db.TopicModel.id_ == id_)
        self._validate_topic_model_finished_training(topic_mdl)

        keywords_per_topic = self._get_keywords_per_topic(topic_mdl)
        examples_per_topic = self._get_examples_per_topic(topic_mdl)

        assert len(keywords_per_topic) == len(examples_per_topic)

        topic_mdl_status_json = self._topic_model_status_json(topic_mdl)
        topic_preview_json = TopicModelPreviewJson(
            {
                "topic_model_id": topic_mdl_status_json["topic_model_id"],
                "topic_model_name": topic_mdl_status_json["topic_model_name"],
                "num_topics": topic_mdl_status_json["num_topics"],
                "topic_names": topic_mdl_status_json["topic_names"],
                "status": topic_mdl_status_json["status"],
                "topic_previews": [
                    OneTopicPreviewJson({"examples": examples, "keywords": keywords})
                    for examples, keywords in zip(
                        examples_per_topic, keywords_per_topic
                    )
                ],
            }
        )
        return topic_preview_json

    @staticmethod
    def _get_keywords_per_topic(topic_mdl: db.TopicModel) -> T.List[T.List[str]]:
        """

        Returns:
            keywords_per_topic: A list of lists of strings.
                List i contains keywords that have highest emission probability.
                under topic i.
        """

        # Look at the documentation at utils.Files.topic_model_keywords_file() for
        # what the file is supposed to look like.
        keywords_file_path = utils.Files.topic_model_keywords_file(topic_mdl.id_)

        keywords_df = pd.read_excel(keywords_file_path, index_col=0, header=0)
        keywords_df = keywords_df.iloc[:-1]  # Remove the "probabilities" row
        return keywords_df.T.values.tolist()  # type: ignore[no-any-return]

    @staticmethod
    def _get_examples_per_topic(topic_mdl: db.TopicModel) -> T.List[T.List[str]]:
        """

        Returns;
            topic_most_likely_examples: A list of list of strings.
                List i within this list contains examples whose most likely topic was
                determined to be topic i.

                i starts counting from zero. The maximum number of examples is determined
                by utils.MAX_NUM_EXAMPLES_PER_TOPIC_IN_PREIVEW
        """

        # Look at the documentation at utils.Files.topic_model_topics_by_doc_file() for
        # what the file is supposed to look like.
        topics_by_doc_path = utils.Files.topic_model_topics_by_doc_file(topic_mdl.id_)
        topics_by_doc_df = pd.read_excel(topics_by_doc_path, index_col=0, header=0)
        bool_mask_topic_most_likely_examples: T.List[pd.Series] = [
            topics_by_doc_df[utils.MOST_LIKELY_TOPIC_COL] == topic_num
            for topic_num in range(topic_mdl.num_topics)
        ]
        examples_per_topic: T.List[T.List[str]] = [
            topics_by_doc_df.loc[bool_mask, utils.CONTENT_COL][
                : utils.MAX_NUM_EXAMPLES_PER_TOPIC_IN_PREIVEW
            ]
            .to_numpy()
            .tolist()
            for bool_mask in bool_mask_topic_most_likely_examples
        ]

        return examples_per_topic


def create_app(
    project_data_dir: Path = Path("./project_data"),
    transformers_cache_dir: Path = Path("./transformers_cache_dir"),
    do_tasks_sychronously: bool = False,
) -> Flask:
    """App factory to for easier testing.

    Args:
        project_data_dir: 
        transforemrs_cache_dir:
        do_tasks_sychronously: Whether to do things like classifier training and LDA
            topic modeling synchronously. This is used to support unit testing.

    Sets:
        app.config["PROJECT_DATA_DIR"]
        app.config["TRANSFORMERS_CACHE_DIR"]
        app.config["SCHEDULER"]
        app.config["MALLET_BIN_DIRECTORY"]

    Returns:
        app: Flask() object.
    """
    app = Flask(__name__, static_url_path="/", static_folder="../../frontend")

    @app.before_request
    def _db_connect() -> None:
        """Ensures that a connection is opened to handle queries by the request."""
        db.DATABASE.connect()

    @app.teardown_request
    def _db_close(exc: T.Optional[Exception]) -> None:
        """Close on tear down."""
        if not db.DATABASE.is_closed():
            db.DATABASE.close()

    app.config["PROJECT_DATA_DIR"] = project_data_dir
    app.config["SCHEDULER"] = Scheduler(do_tasks_sychronously=do_tasks_sychronously)
    app.config["TRANSFORMERS_CACHE_DIR"] = transformers_cache_dir

    mallet_bin_dir = os.environ.get("MALLET_BIN_DIRECTORY", None)
    if mallet_bin_dir is None:
        print(
            "Please set the MALLET_BIN_DIRECTORY environment variable. Have a look at the README for why."
        )
        sys.exit(1)
    app.config["MALLET_BIN_DIRECTORY"] = mallet_bin_dir

    api = Api(app)
    # `utils.Files` uses flask.current_app. Since we're not
    # handling a request just yet, we need this.
    with app.app_context():
        # Create the project data directory
        # In the future, this hould be disabled.
        utils.Files.project_data_dir(ensure_exists=True)
        utils.Files.supervised_dir(ensure_exists=True)
        utils.Files.unsupervised_dir(ensure_exists=True)

    lsresource_cls: T.Tuple[T.Type[BaseResource], ...] = (
        Classifiers,
        OneClassifier,
        ClassifiersTrainingFile,
        ClassifiersTestSets,
        OneClassifierTestSet,
        ClassifiersTestSetsFile,
        TopicModels,
        OneTopicModel,
        TopicModelsTrainingFile,
        TopicModelsTopicsNames,
        TopicModelsTopicsPreview,
    )
    for resource_cls in lsresource_cls:
        assert (
            resource_cls.url[0] == "/"
        ), f"{resource_cls.__name__}.url must start with a /"
        url = API_URL_PREFIX + resource_cls.url
        api.add_resource(resource_cls, url)

    return app
