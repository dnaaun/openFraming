import io
import shutil
import unittest
from unittest import mock

from tests.common import AppMixin
from tests.common import debug_on
from tests.common import TESTING_FILES_DIR

from flask_app import db
from flask_app import utils
from flask_app.app import API_URL_PREFIX
from flask_app.app import ClassifierStatusJson
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
    def test_get(self) -> None:
        url = API_URL_PREFIX + "/classifiers/"
        with self._app.test_client() as client, self._app.app_context():
            resp = client.get(url)
            self._assert_response_success(resp, url)
            resp_json = resp.get_json()
            self.assertIsInstance(resp_json, list)

            clsf_status = resp_json[0]
            expected_classifier_status = ClassifierStatusJson(
                classifier_id=self._clsf.classifier_id,
                classifier_name=self._clsf.name,
                category_names=self._clsf.category_names,
                status="not_begun",
                trained_by_openFraming=False,
                metrics=None,
            )
            self.assertDictEqual(clsf_status, dict(expected_classifier_status))

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

            # Assert response json
            expected_classifier_status = ClassifierStatusJson(
                classifier_id=self._clsf.classifier_id,
                classifier_name=self._clsf.name,
                category_names=self._clsf.category_names,
                status="training",
                trained_by_openFraming=False,
                metrics=None,
            )
            clsf_status = res.get_json()
            self.assertDictEqual(clsf_status, dict(expected_classifier_status))

        # Assert shceduler called
        scheduler.add_classifier_training.assert_called_with(
            classifier_id=self._clsf.classifier_id,
            labels=self._clsf.category_names,
            model_path=utils.TRANSFORMERS_MODEL,
            dev_file=str(dev_set_file),
            train_file=str(train_set_file),
            cache_dir=str(self._app.config["TRANSFORMERS_CACHE_DIR"]),
            output_dir=str(output_dir),
        )

        # Assert files created
        self.assertTrue(dev_set_file.exists())
        self.assertTrue(train_set_file.exists())

    @debug_on()
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
        self._clsf.dev_set.save()
        self._clsf.train_set.save()
        self._clsf.save()

        scheduler: Scheduler = self._app.config["SCHEDULER"]
        scheduler.add_classifier_training(
            classifier_id=self._clsf.classifier_id,
            labels=self._clsf.category_names,
            model_path=utils.TRANSFORMERS_MODEL,
            train_file=str(train_set_file),
            dev_file=str(dev_set_file),
            cache_dir=self._app.config["TRANSFORMERS_CACHE_DIR"],
            output_dir=str(output_dir),
        )

        expected_classifier_status = dict(
            classifier_id=self._clsf.classifier_id,
            classifier_name=self._clsf.name,
            category_names=self._clsf.category_names,
            status="completed",
            trained_by_openFraming=False,
            # metrics is missing, on purpose
        )

        url = API_URL_PREFIX + "/classifiers/"
        with self._app.test_client() as client, self._app.app_context():
            resp = client.get(url)
        self._assert_response_success(resp, url)

        resp_json = resp.get_json()
        assert isinstance(resp_json, list)
        clsf_status = resp_json[0]

        metrics = clsf_status.pop("metrics")
        self.assertDictEqual(clsf_status, expected_classifier_status)

        self.assertSetEqual(
            {"macro_f1_score", "accuracy", "macro_precision", "macro_recall",},
            set(metrics.keys()),
        )

        self.assertSetEqual(set(map(type, metrics.values())), {float})


if __name__ == "__main__":
    unittest.main()
