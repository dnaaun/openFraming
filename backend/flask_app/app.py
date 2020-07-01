"""All the flask api endpoints."""
import csv
import io
import logging
import typing as T
from collections import Counter
from pathlib import Path

from flask import current_app
from flask import Flask
from flask_restful import Api  # type: ignore
from flask_restful import reqparse
from flask_restful import Resource
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


class ClassifierRelatedResource(BaseResource):
    """Base class to define utility functions related to classifiers."""

    @staticmethod
    def _classifier_status(clsf: db.Classifier) -> utils.Json:
        """Process a Classifier instance and format it into the API spec.

        Returns:
            {
                "classifier_id": int,
                "name": str,
                "trained_by_openFraming": bool,
                "category_names": T.List[str],
                "training_status": T.Union["not_begun", "training", "completed"]
          }
        """
        if clsf.train_set is None:
            training_status = "not_begun"
        else:
            if clsf.train_set.training_or_inference_completed:
                training_status = "completed"
            else:
                training_status = "training"

        category_names = clsf.category_names

        return {
            "classifier_id": clsf.classifier_id,
            "name": clsf.name,
            "trained_by_openFraming": clsf.trained_by_openFraming,
            "category_names": category_names,
            "training_status": training_status,
        }


class Classifiers(ClassifierRelatedResource):
    """Create a classifer, get a list of classifiers."""

    url = "/classifiers/"

    def __init__(self) -> None:
        """Set up request parser."""
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument(name="name", type=str, required=True)

        def category_names_type(val: T.Any) -> str:
            if not isinstance(val, str):
                raise ValueError("must be str")
            if "," in val:
                raise ValueError("can't contain commas.")

            return val

        self.reqparse.add_argument(
            name="category_names",
            type=category_names_type,
            action="append",
            required=True,
            help="",
        )

    def post(self) -> utils.Json:
        """Create a classifier.

        req_body:
            json:
                {
                    "name": str,
                    "category_names": T.List[str]
                }

        Returns:
            {
                "classifier_id": int,
                "name": str,
                "trained_by_openFraming": bool,
                "status": "not_begun",
                "category_names": T.List[str],
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

    def get(self) -> utils.Json:
        """Get a list of classifiers.

        Returns:
            [
              {
                "classifier_id": int,
                "name": str,
                "trained_by_openFraming": bool,
                "status": T.Union["not_begun", "training", "completed"]
                "category_names": T.List[str],
              },
              ...
            ]
        """
        res: T.List[utils.Json] = [
            self._classifier_status(clsf) for clsf in db.Classifier.select()
        ]
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

    def post(self, classifier_id: int) -> utils.Json:
        """Upload a training set for classifier, and start training.

        Body:
            FormData: with "file" item.

        Returns:
            {
                "classifier_id": int,
                "name": str,
                "trained_by_openFraming": bool,
                "category_names": T.List[str],
                "training_status": "training"
            }

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
            labels=classifier.category_names,
            model_path=utils.TRANSFORMERS_MODEL,
            data_dir=str(utils.Files.classifier_dir(classifier_id)),
            cache_dir=current_app.config["TRANSFORMERS_CACHE_DIR"],
            output_dir=str(
                utils.Files.classifier_output_dir(classifier_id, ensure_exists=True)
            ),
        )

        return self._classifier_status(classifier)

    @staticmethod
    def _validate_training_file_and_get_data(
        category_names: T.List[str], file_: FileStorage
    ) -> T.Tuple[T.List[str], T.List[T.List[str]]]:
        """Validate user input and return uploaded CSV data, without the headers.

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


TopicModelStatus = TypedDict(
    "TopicModelStatus",
    {
        "topic_model_id": int,
        "topic_model_name": str,
        "num_topics": int,
        "topic_names": T.Optional[T.List[str]],
        "status": T.Literal["not_begun", "training", "topics_to_be_named", "completed"],
        # TODO: Update backend README to reflect API change for line above.
    },
)


class TopicModelRelatedResource(BaseResource):
    """Base class to define utility functions related to classifiers."""

    @staticmethod
    def _topic_model_status(topic_mdl: db.TopicModel) -> TopicModelStatus:
        topic_names = topic_mdl.topic_names
        status: T.Literal["not_begun", "training", "topics_to_be_named", "completed"]
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

        return TopicModelStatus(
            topic_model_name=topic_mdl.name,
            topic_model_id=topic_mdl.id_,
            num_topics=topic_mdl.num_topics,
            topic_names=topic_names,
            status=status,
        )


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

    def post(self) -> TopicModelStatus:
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
        return self._topic_model_status(topic_mdl)

    def get(self) -> T.List[TopicModelStatus]:
        res = [
            self._topic_model_status(topic_mdl) for topic_mdl in db.TopicModel.select()
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

    def post(self, id_: int) -> TopicModelStatus:
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
            training_file=str(train_file),
            num_topics=topic_mdl.num_topics,
            fname_keywords=str(utils.Files.topic_model_keywords_file(id_)),
            fname_topics_by_doc=str(
                utils.Files.topic_model_probabilities_by_example_file(id_)
            ),
        )
        topic_mdl.lda_set = db.LDASet()
        topic_mdl.lda_set.save()
        topic_mdl.save()

        # Refresh classifier
        topic_mdl = db.TopicModel.get(db.TopicModel.id_ == id_)

        # model_scheduler: ModelScheduler = current_app.config["SCHEDULER"]

        return self._topic_model_status(topic_mdl)

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

    Returns:
        app: Flask() object.
    """
    app = Flask(__name__, static_url_path="/", static_folder="../frontend")

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
        ClassifiersTrainingFile,
        TopicModels,
        TopicModelsTrainingFile,
    )
    for resource_cls in lsresource_cls:
        assert (
            resource_cls.url[0] == "/"
        ), f"{resource_cls.__name__}.url must start with a /"
        url = API_URL_PREFIX + resource_cls.url
        api.add_resource(resource_cls, url)

    return app
