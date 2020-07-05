import csv
import io
import shutil
import unittest
from unittest import mock

from tests.common import AppMixin
from tests.common import make_csv_file
from tests.common import TESTING_FILES_DIR

from flask_app import db
from flask_app import utils
from flask_app.app import API_URL_PREFIX
from flask_app.app import ClassifierStatusJson
from flask_app.app import ClassifierTestSetStatusJson
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

    def test_get_one_classifier(self) -> None:
        url = API_URL_PREFIX + f"/classifiers/{self._clsf.classifier_id}"
        with self._app.test_client() as client, self._app.app_context():
            resp = client.get(url)
            self._assert_response_success(resp, url)
            resp_json = resp.get_json()
            self.assertIsInstance(resp_json, dict)

            clsf_status = resp_json
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

    def test_training_and_testing(self) -> None:
        with self.subTest("training classifier"):
            with self._app.app_context():
                output_dir = utils.Files.classifier_output_dir(self._clsf.classifier_id)
                dev_set_file = utils.Files.classifier_dev_set_file(
                    self._clsf.classifier_id
                )
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
                cache_dir=str(self._app.config["TRANSFORMERS_CACHE_DIR"]),
                output_dir=str(output_dir),
                num_train_epochs=1.0,
            )

            expected_classifier_status = dict(
                classifier_id=self._clsf.classifier_id,
                classifier_name=self._clsf.name,
                category_names=self._clsf.category_names,
                status="completed",
                trained_by_openFraming=False,
                # metrics is missing, on purpose
            )

            file_upload_url = API_URL_PREFIX + "/classifiers/"
            with self._app.test_client() as client, self._app.app_context():
                resp = client.get(file_upload_url)
            self._assert_response_success(resp, file_upload_url)

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

        with self.subTest("doing test set prediction"):
            main_test_sets_url = (
                API_URL_PREFIX + f"/classifiers/{self._clsf.classifier_id}/test_sets/"
            )

            test_set_name = "my first test set ever!"
            req_json = {"test_set_name": test_set_name}

            with self._app.test_client() as client, self._app.app_context():
                resp = client.post(main_test_sets_url, json=req_json)

                # Assert successful response
                self._assert_response_success(resp, file_upload_url)

                # Assert test set created in db.
                # .get() should raise an error if nothing was created.
                created_test_set = self._clsf.test_sets.get()  # type: ignore

                # Assert the right response was returned
                resp_json = resp.get_json()
                expected_json = ClassifierTestSetStatusJson(
                    test_set_id=created_test_set.id_,
                    classifier_id=self._clsf.classifier_id,
                    test_set_name=test_set_name,
                    status="not_begun",
                )
                self.assertDictEqual(resp_json, dict(expected_json))

                # Test the single-entity endpoint
                single_test_set_url = (
                    API_URL_PREFIX
                    + f"/classifiers/{self._clsf.classifier_id}/test_sets/{created_test_set.id_}"
                )

                one_test_set_resp = client.get(single_test_set_url)
                self._assert_response_success(one_test_set_resp, single_test_set_url)
                self.assertDictEqual(one_test_set_resp.get_json(), dict(expected_json))

        with self.subTest("uploading a test set and running inference"):
            file_upload_url = (
                API_URL_PREFIX
                + f"/classifiers/{self._clsf.classifier_id}/test_sets/{created_test_set.id_}/file"
            )
            valid_test_file_table = [
                [f"{utils.CONTENT_COL}"],
                ["galaxies"],
                ["ocean"],
                ["directions?"],
            ]

            file_to_upload = make_csv_file(valid_test_file_table)
            with self._app.test_client() as client, self._app.app_context():
                resp = client.post(
                    file_upload_url, data={"file": (file_to_upload, "test.csv")}
                )
                self._assert_response_success(resp, file_upload_url)

                # Assert status changed to predicting
                expected_json = resp.get_json()
                self.assertEqual(expected_json.get("status"), "predicting")

                # Assert file was created
                test_set_file = utils.Files.classifier_test_set_file(
                    self._clsf.classifier_id, created_test_set.id_
                )
                self.assertTrue(test_set_file.exists())

                # Assert the test results
                test_set_predictions_file = utils.Files.classifier_test_set_predictions_file(
                    self._clsf.classifier_id, created_test_set.id_
                )
                self.assertTrue(test_set_predictions_file.exists())

                # Assert test set results make sense
                with test_set_predictions_file.open() as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                self.assertListEqual(
                    rows[0], [f"{utils.CONTENT_COL}", f"{utils.PREDICTED_LABEL_COL}",]
                )
                examples, predicted_labels = zip(*rows[1:])
                (expected_examples,) = zip(*valid_test_file_table[1:])
                self.assertSequenceEqual(examples, expected_examples)
                self.assertTrue(set(predicted_labels) <= {"up", "down"})

                # Assert the test set is reported as "completed" now
                resp = client.get(main_test_sets_url)
                self._assert_response_success(resp, main_test_sets_url)

                resp_json = resp.get_json()
                self.assertEqual(resp_json[0]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
