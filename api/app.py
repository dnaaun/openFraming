"""All the flask api endpoints."""
import abc
import typing as T

from flask import Flask
from flask_restful import Api  # type: ignore
from flask_restful import reqparse
from flask_restful import Resource

Json = T.Dict[str, T.Any]

app = Flask(__name__)
api = Api(app)


parser = reqparse.RequestParser()
args = parser.parse_args()


class BaseResource(Resource, abc.ABC):
    """Used to make sure that subclasses have a .url attribute.

    Attributes:
        url:
    """

    url: str


class ClassifierList(BaseResource):
    """Create a classifer, get a list of classifiers."""

    url = "/classifiers"

    def post(self) -> Json:
        """Create a classifier.

        req_body:
            json:
                {
                    "name": str,
                    "category_names": [str, ...]
                }
        """
        raise NotImplementedError()

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
        raise NotImplementedError()


class ClassifierProgress(BaseResource):
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


_RESOURCES: T.List[T.Type[BaseResource]] = [ClassifierList, ClassifierProgress]


def main() -> None:
    """Add the resource classes with api.add_resource."""
    url_prefix = "/"

    for resource_cls in _RESOURCES:
        assert (
            resource_cls.url[0] == "/"
        ), f"{resource_cls.__name__}.url must start with a /"
        api.add_resource(
            resource_cls,
            url_prefix + resource_cls.url + "classifiers/<int:classifier_id>/progress",
        )


if __name__ == "__main__":
    main()
