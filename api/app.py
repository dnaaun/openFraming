import typing as T

from flask import Flask
from flask_restful import Api  # type: ignore
from flask_restful import Resource

app = Flask(__name__)
api = Api(app)

Json = T.Dict[str, T.Any]


class ClassifierList(Resource):
    def post(self) -> Json:
        return {}

    def get(self) -> Json:
        return {"json_key": "json value is hello world"}


class ClassifierBase(Resource):
    pass


class ClassifierProgress(ClassifierBase):
    def get(self, classifier_id: int = 20) -> Json:

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


_PREFIX = "/"  # For now, we'll change it later.
api.add_resource(
    ClassifierProgress,
    _PREFIX + "classifiers/<int:classifier_id>/progress",
    endpoint="classifiers",
)
