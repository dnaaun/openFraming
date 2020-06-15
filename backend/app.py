"""All the flask api endpoints."""
import typing as T

from flask import Flask
from flask_restful import Api  # type: ignore
from flask_restful import reqparse
from flask_restful import Resource

import db

# mypy doesn't support recrsive types, so this is the best we can do
Json = T.Optional[T.Union[T.List[T.Any], T.Dict[str, T.Any], int, str, bool]]

app = Flask(__name__, static_url_path="/", static_folder="../frontend")
api = Api(app,)


class BaseResource(Resource):
    """Used to make sure that subclasses have a .url attribute.

    Attributes:
        url:
    """

    url: str


class ClassifiersTrainingData(BaseResource):
    """Create a classifer, get a list of classifiers."""

    url = "/classifiers/<int:classifier_id/training_data"

    def __init__(self) -> None:
        """Set up request parser."""
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument(name="name", type=str, required=True)

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
        if len(args["category_names"]) < 2:
            return {"message": {"category_names": "need at least 2 categories."}}

        category_names = ",".join(args["category_names"])
        name = args["name"]
        db.Classifier.create(name=name, category_names=category_names)
        return None

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
        res: T.List[Json] = list(db.Classifier.select().dicts())
        return res


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

            print("val is", val)
            return val

        self.reqparse.add_argument(
            name="category_names",
            type=category_names_type,
            action="append",
            required=True,
            help="",
        )

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
        if len(args["category_names"]) < 2:
            return {"message": {"category_names": "need at least 2 categories."}}

        category_names = ",".join(args["category_names"])
        name = args["name"]
        db.Classifier.create(name=name, category_names=category_names)
        return None

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
        res: T.List[Json] = list(db.Classifier.select().dicts())
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


_RESOURCES: T.List[T.Type[BaseResource]] = [Classifiers, ClassifiersProgress]


def main() -> None:
    """Add the resource classes with api.add_resource."""
    url_prefix = "/api"

    for resource_cls in _RESOURCES:
        assert (
            resource_cls.url[0] == "/"
        ), f"{resource_cls.__name__}.url must start with a /"
        url = url_prefix + resource_cls.url
        api.add_resource(resource_cls, url)


main()
