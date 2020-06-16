"""All the flask api endpoints."""
import logging
import os
import typing as T
from pathlib import Path

from flask import Flask
from flask_restful import Api  # type: ignore
from flask_restful import reqparse
from flask_restful import Resource
from werkzeug.exceptions import BadRequest

import db

logger = logging.getLogger("__main__")

PROJECT_DATA_DIR = Path(os.environ.get("PROJECT_DATA_DIR", "./project_data"))
"""The data directory under which model weights, training files, and predictions
will be stored."""

SUPERVISED_DATA_DIR = PROJECT_DATA_DIR / "supervised"
UNSUPERVISED_DATA_DIR = PROJECT_DATA_DIR / "unsupervised"

# mypy doesn't support recrsive types, so this is the best we can do
Json = T.Optional[T.Union[T.List[T.Any], T.Dict[str, T.Any], int, str, bool]]

app = Flask(__name__, static_url_path="/", static_folder="../frontend")
api = Api(app)


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
        training_completed = (
            # TODO: Will probably have to change once we actually upload a training
            # set
            clsf.training_set is not None
            and clsf.training_set.training_or_inference_completed
        )  # Training has completed

        return {
            "classifier_id": clsf.classifier_id,
            "name": clsf.name,
            "trained_by_openFraming": clsf.trained_by_openFraming,
            "training_completed": training_completed,
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


class ClassifiersProgress(BaseResource):
    """Get classsifier progress."""

    url = "/classifiers/<int:classifier_id>/progress"

    def get(self, classifier_id: int = 20) -> Json:
        """Get clf progress.

        Returns:
            One_of["training_model", "running_inference_on_dev_set", "done"]
        """
        progress = 50
        if classifier_id == 0:
            stage = "training_model"
        elif classifier_id == 1:
            stage = "running_inference_on_dev_set"
        elif classifier_id == 2:
            stage = "done"
            progress = 100
        else:
            raise Exception("classifier_id provided doesn't exist.")

        return {"progress": progress, "stage": stage}


def main() -> None:
    """Add the resource classes with api.add_resource."""
    url_prefix = "/api"

    # Create the project data directory
    # In the future, this hould be disabled.
    if not PROJECT_DATA_DIR.exists():
        logger.warning("Creating PROJECT_DATA_DIR because it doesn't exist.")
        PROJECT_DATA_DIR.mkdir()
    else:
        if not SUPERVISED_DATA_DIR.exists():
            logger.warning("Creating SUPERVISED_DATA_DIR because it doesn't exist.")
            SUPERVISED_DATA_DIR.mkdir()
        if not UNSUPERVISED_DATA_DIR.exists():
            logger.warning("Creating UNSUPERVISED_DATA_DIR because it doesn't exist.")
            UNSUPERVISED_DATA_DIR.mkdir()

    for resource_cls in BaseResource.__subclasses__():
        assert (
            resource_cls.url[0] == "/"
        ), f"{resource_cls.__name__}.url must start with a /"
        url = url_prefix + resource_cls.url
        api.add_resource(resource_cls, url)


main()
