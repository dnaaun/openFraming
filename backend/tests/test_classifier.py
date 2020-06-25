import functools
import io
import pdb
import shutil
import sys
import tempfile
import traceback
import typing as T
import unittest
from pathlib import Path
from unittest import mock

import peewee as pw
from flask import Response

from flask_app import db
from flask_app import utils
from flask_app.app import API_URL_PREFIX
from flask_app.app import create_app
from flask_app.modeling.train_queue import ModelScheduler

F = T.TypeVar("F", bound=T.Callable[[T.Any], T.Any])


def debug_on(*exceptions: T.Type[Exception]) -> T.Callable[[F], F]:
    if not exceptions:
        exceptions = (Exception,)

    def decorator(f: F) -> F:
        @functools.wraps(f)
        def wrapper(*args: T.Any, **kwargs: T.Any) -> T.Any:
            try:
                return f(*args, **kwargs)  # type: ignore
            except exceptions:
                info = sys.exc_info()
                traceback.print_exception(*info)
                pdb.post_mortem(info[2])

        return wrapper  # type: ignore

    return decorator


class TestMixin:
    """A base class for all unittest mixins.

    Allows having a bunch of mixins and calling super().setUp() once.

    Actually, I'm not sure this is needed. # TODO:
    """

    def setUp(self) -> None:
        super().setUp()  # type: ignore

    def tearDown(self) -> None:
        super().tearDown()  # type: ignore


class AppMixin(TestMixin):
    def setUp(self) -> None:
        self._temp_proj_dir = Path(tempfile.mkdtemp())
        self._test_db = pw.SqliteDatabase(":memory:")
        self._app = create_app(project_data_dir=self._temp_proj_dir)

        # Bind model to test db, since we have a complete list of all models, we do not need
        # to recursively bind dependencies.
        # http://docs.peewee-orm.com/en/latest/peewee/database.html?highlight=bind#testing-peewee-applications
        self._test_db.bind(db.MODELS, bind_refs=False, bind_backrefs=False)
        self._test_db.connect()
        self._test_db.create_tables(db.MODELS)
        super().setUp()

    def tearDown(self) -> None:
        self._test_db.drop_tables(db.MODELS)
        self._test_db.close()
        shutil.rmtree(self._temp_proj_dir)


class UrlTestingMixin(TestMixin):
    @staticmethod
    def _assert_response_success(res: Response) -> None:
        if res.status_code != 200:
            raise AssertionError(
                f"request failed with status: {res.status_code}." f"data: {res.data}"
            )


class TestClassifiers(AppMixin, UrlTestingMixin, unittest.TestCase):
    def test_start_training(self) -> None:
        # Mock the scheduler
        scheduler: ModelScheduler = self._app.config["MODEL_SCHEDULER"]
        scheduler.add_training_process: mock.MagicMock = mock.MagicMock(return_value=None)  # type: ignore

        # Create a classifer in the database
        clsf = db.Classifier.create(
            name="test_classifier", category_names=["up", "down"]
        )
        utils.Files.classifier_dir(clsf.classifier_id, ensure_exists=True)

        valid_training_contents = "\n".join(
            [
                f"{utils.LABELLED_CSV_CONTENT_COL},{utils.LABELLED_CSV_LABEL_COL}",
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
            scheduler.add_training_process.assert_called_with(
                labels=clsf.category_names,
                model_path=utils.TRANSFORMERS_MODEL,
                data_dir=str(utils.Files.classifier_dir(clsf.classifier_id)),
                cache_dir=self._app.config["TRANSFORMERS_CACHE_DIR"],
                output_dir=str(utils.Files.classifier_output_dir(clsf.classifier_id)),
            )


if __name__ == "__main__":
    unittest.main()
