"""All the flask api endpoints."""
import logging
import os
import typing as T
from functools import lru_cache
from pathlib import Path

from flask import current_app
from flask import Flask
from flask_restful import Api  # type: ignore
from flask_restful import reqparse
from flask_restful import Resource
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import BadRequest
from werkzeug.exceptions import NotFound

import db

API_URL_PREFIX = "/api"

logger = logging.getLogger("__main__")
# mypy doesn't support recrsive types, so this is the best we can do
Json = T.Optional[T.Union[T.List[T.Any], T.Dict[str, T.Any], int, str, bool]]


class Files:
    """A class for defining where files will be stored."""

    @classmethod
    @lru_cache
    def project_data_dir(cls) -> Path:
        """Dir where all project related files will be stored."""
        return current_app.config["PROJECT_DATA_DIR"]  # type: ignore

    @classmethod
    @lru_cache
    def supervised_dir(cls) -> Path:
        """Dir for classifier weights, training and inference data."""
        return cls.project_data_dir() / "supervised"

    @classmethod
    @lru_cache
    def unsupervised_dir(cls) -> Path:
        """Dir for LDA results, training and inference data."""
        return cls.project_data_dir() / "unsupervised"


class BaseResource(Resource):
    """Used to make sure that subclasses have a .url attribute.

    Attributes:
        url:
    """

    url: str


class Classifiers(BaseResource):
    """Create a classifer, get a list of classifiers."""

    url = "/classifiers"

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

    @staticmethod
    def _classifier_status(clsf: db.Classifier) -> Json:
        """Process a Classifier instance and format it into the API spec.

        Returns:
            {
                "classifier_id": int,
                "name": str,
                "trained_by_openFraming": bool,
                "training_completed": bool
          }
        """
        if clsf.training_set is None:
            status = "not_begun"
        else:
            if clsf.training_set.training_or_inference_completed:
                status = "completed"
            else:
                status = "training"

        return {
            "classifier_id": clsf.classifier_id,
            "name": clsf.name,
            "trained_by_openFraming": clsf.trained_by_openFraming,
            "status": status,
        }

    def post(self) -> Json:
        """Create a classifier.

        req_body:
            json:
                {
                    "name": str,
                    "category_names": [str, ...]
                }
        """
        args = self.reqparse.parse_args()
        if (
            len(args["category_names"]) < 2
        ):  # I don't know how to do this validation in the
            # RequestParser
            raise BadRequest("must be at least two categories.")

        category_names = ",".join(args["category_names"])
        name = args["name"]
        # Use a placeholder for file_path to get the auto incremented id
        clsf = db.Classifier.create(
            name=name, category_names=category_names, dir_path="WILL_BE_REPLACED"
        )
        dir_path = f"classifier_{clsf.classifier_id}"
        clsf.dir_path = dir_path
        clsf.save()
        return self._classifier_status(clsf)

    def get(self) -> Json:
        """Get a list of classifiers.

        Returns:
            [
              {
                "classifier_id": int,
                "name": str,
                "trained_by_openFraming": bool,
                "training_completed": bool
              },
              ...
            ]
        """
        res: T.List[Json] = [
            self._classifier_status(clsf) for clsf in db.Classifier.select()
        ]
        return res


class ClassifierTrainingFile(BaseResource):
    """Upload training data to the classifier."""

    url = "/classifiers/<int:classifier_id>/training/file"

    def __init__(self) -> None:
        """Set up request parser."""
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument(
            name="file", type=FileStorage, required=True, location="files"
        )

    @staticmethod
    def _training_data_is_valid(clsf: db.Classifier, data: T.List[T.List[str]]) -> bool:
        """Check if the data indicated is a valid training file for clsf.

        Check that every "sentence" and "category" column is non empty, no category
        contains a comma, and that the set of unique categories matches
        clsf.category_names.
        """

    def post(self) -> Json:
        """."""

    def get(self) -> Json:
        """."""


def create_app() -> Flask:
    """App factory to avoid forcibly creating an app object at import time.

    Useful for testing.
    """
    app = Flask(__name__, static_url_path="/", static_folder="../frontend")
    app.config["PROJECT_DATA_DIR"] = Path(
        os.environ.get("PROJECT_DATA_DIR", "./project_data")
    )
    api = Api(app)

    # `Directories` uses flask.current_app. Since we're not
    # handling a request just yet, we need this.
    with app.app_context():
        # Create the project data directory
        # In the future, this hould be disabled.
        for dir_ in [
            Files.project_data_dir(),
            Files.supervised_dir(),
            Files.unsupervised_dir(),
        ]:
            if not dir_.exists():
                logger.warning(f"Creating {str(dir_)} because it doesn't exist.")

    for resource_cls in BaseResource.__subclasses__():
        assert (
            resource_cls.url[0] == "/"
        ), f"{resource_cls.__name__}.url must start with a /"
        url = API_URL_PREFIX + resource_cls.url
        api.add_resource(resource_cls, url)

    return app
