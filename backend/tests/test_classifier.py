import io
import shutil
import unittest
from unittest import mock

from tests.common import AppMixin
from tests.common import TESTING_FILES_DIR

from flask_app import db
from flask_app import utils
from flask_app.app import API_URL_PREFIX
from flask_app.modeling.train_queue import Scheduler


class ClassifierMixin(AppMixin):
    def setUp(self) -> None:
        super().setUp()
        with self._app.app_context():

            # Create a classifer in the database
            self._clsf = db.Classifier.create(
                name="test_classifier", category_names=["up", "down"]
            )
            utils.Files.classifier_dir(self._clsf.classifier_id, ensure_exists=True)

        self._valid_training_contents = "\n".join(
            [
                f"{utils.CONTENT_COL},{utils.LABEL_COL}",
                "sky,up",
                "earth,down",
                "dimonds,down",
                "stars,up",
                "dead sea,down",
                "moon,up",
            ]
        )


class TestClassifiers(ClassifierMixin, unittest.TestCase):
    def test_trigger_training(self) -> None:
        # Mock the scheduler
        scheduler: Scheduler = self._app.config["SCHEDULER"]
        scheduler.add_classifier_training: mock.MagicMock = mock.MagicMock(return_value=None)  # type: ignore

        test_url = (
            API_URL_PREFIX + f"/classifiers/{self._clsf.classifier_id}/training/file"
        )
        file_ = io.BytesIO(self._valid_training_contents.encode())
        with self._app.test_client() as client, self._app.app_context():
            output_dir = utils.Files.classifier_output_dir(self._clsf.classifier_id)
            dev_set_file = utils.Files.classifier_dev_set_file(self._clsf.classifier_id)
            train_set_file = utils.Files.classifier_train_set_file(
                self._clsf.classifier_id
            )

            res = client.post(test_url, data={"file": (file_, "labeled.csv")},)
            self._assert_response_success(res)
        scheduler.add_classifier_training.assert_called_with(
            labels=self._clsf.category_names,
            model_path=utils.TRANSFORMERS_MODEL,
            dev_file=str(dev_set_file),
            train_file=str(train_set_file),
            cache_dir=self._app.config["TRANSFORMERS_CACHE_DIR"],
            output_dir=str(output_dir),
        )

        self.assertTrue(dev_set_file.exists())
        self.assertTrue(train_set_file.exists())

    def test_actual_training(self) -> None:
        with self._app.app_context():
            output_dir = utils.Files.classifier_output_dir(self._clsf.classifier_id)
            dev_set_file = utils.Files.classifier_dev_set_file(self._clsf.classifier_id)
            train_set_file = utils.Files.classifier_train_set_file(
                self._clsf.classifier_id
            )

        # Copy over the files
        shutil.copy(TESTING_FILES_DIR / "classifiers" / "dev.csv", dev_set_file)
        shutil.copy(TESTING_FILES_DIR / "classifiers" / "train.csv", train_set_file)

        # Update the database
        self._clsf.train_set = db.LabeledSet()
        self._clsf.dev_set = db.LabeledSet()
        self._clsf.save()

        scheduler: Scheduler = self._app.config["SCHEDULER"]
        scheduler.add_classifier_training(
            labels=self._clsf.category_names,
            model_path=utils.TRANSFORMERS_MODEL,
            train_file=str(train_set_file),
            dev_file=str(dev_set_file),
            cache_dir=self._app.config["TRANSFORMERS_CACHE_DIR"],
            output_dir=str(output_dir),
        )


if __name__ == "__main__":
    unittest.main()
