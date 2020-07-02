import io
import unittest
from unittest import mock

from tests.common import AppMixin

from flask_app import db
from flask_app import utils
from flask_app.app import API_URL_PREFIX
from flask_app.modeling.train_queue import Scheduler


class TestClassifiers(AppMixin, unittest.TestCase):
    def test_start_training(self) -> None:
        # Mock the scheduler
        with self._app.app_context():
            scheduler: Scheduler = self._app.config["SCHEDULER"]
            scheduler.add_classifier_training: mock.MagicMock = mock.MagicMock(return_value=None)  # type: ignore

            # Create a classifer in the database
            clsf = db.Classifier.create(
                name="test_classifier", category_names=["up", "down"]
            )
            utils.Files.classifier_dir(clsf.classifier_id, ensure_exists=True)

        valid_training_contents = "\n".join(
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
        test_url = API_URL_PREFIX + f"/classifiers/{clsf.classifier_id}/training/file"
        file_ = io.BytesIO(valid_training_contents.encode())
        with self._app.test_client() as client, self._app.app_context():
            res = client.post(test_url, data={"file": (file_, "labeled.csv")},)
            self._assert_response_success(res)
            scheduler.add_classifier_training.assert_called_with(
                labels=clsf.category_names,
                model_path=utils.TRANSFORMERS_MODEL,
                data_dir=str(utils.Files.classifier_dir(clsf.classifier_id)),
                cache_dir=self._app.config["TRANSFORMERS_CACHE_DIR"],
                output_dir=str(utils.Files.classifier_output_dir(clsf.classifier_id)),
            )


if __name__ == "__main__":
    unittest.main()
