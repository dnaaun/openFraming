import csv
import functools
import io
import logging
import os
import pdb  # type: ignore
import shutil
import sys
import tempfile
import traceback
import typing as T
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd  # type: ignore
import typing_extensions as TT
from flask import current_app
from flask import Response
from redis import Redis
from rq import Queue
from rq import Worker

from flask_app import settings
from flask_app.app import create_app
from flask_app.database import models
from flask_app.settings import Settings

F = T.TypeVar("F", bound=T.Callable[..., T.Any])


TESTING_FILES_DIR = Path(__file__).parent / "testing_files"


class MockedException(Exception):
    pass


def exception_raising_mock(*args: T.Any, **kwargs: T.Any) -> None:
    raise MockedException()


def make_csv_file(table: T.List[T.List[str]]) -> io.BytesIO:
    text_io = io.StringIO()
    writer = csv.writer(text_io)
    writer.writerows(table)
    text_io.seek(0)
    csv_file = io.BytesIO(text_io.read().encode())
    return csv_file


def debug_on(*exceptions: T.Type[Exception]) -> T.Callable[[F], F]:
    """Decorator to go to pdb prompt on exceptions."""
    # From stackoverflow.
    if not exceptions:
        exceptions = (Exception,)

    def decorator(f: F) -> F:
        @functools.wraps(f)
        def wrapper(*args: T.Any, **kwargs: T.Any) -> T.Any:
            try:
                return f(*args, **kwargs)
            except exceptions:
                info = sys.exc_info()
                traceback.print_exception(*info)
                pdb.post_mortem(info[2])

        return wrapper  # type: ignore

    return decorator


class AppMixin(unittest.TestCase):
    """Setup Flask() object, mock PROJECT_DATA_DIRECTORY and DATABASE_FILE."""

    def setUp(self) -> None:
        # NOTE: IF you remove the following, make sure to remove the shutil.rmtree call
        # in tearDown() as well.

        super().setUp()
        with mock.patch.dict(
            os.environ,
            {"PROJECT_DATA_DIRECTORY": tempfile.mkdtemp(prefix="project_data_")},
        ):
            app = create_app(logging_level=logging.DEBUG)
        app.config["TESTING"] = True
        app.config["DEBUG"] = True

        # simulate `with app:`

        self._app_context = app.app_context()
        self._app_context.push()

    def tearDown(self) -> None:
        super().tearDown()
        self._app_context.pop()
        models.database_proxy.drop_tables(models.MODELS)
        models.database_proxy.close()
        shutil.rmtree(Settings.PROJECT_DATA_DIRECTORY)

    def _assert_response_success(
        self, res: Response, url: T.Optional[str] = None
    ) -> None:
        if res.status_code != 200:
            raise AssertionError(
                "\n".join(
                    [
                        "Request failed.",
                        f"URL: '{url}'." if url is not None else "",
                        f"Status code: '{res.status_code}'.",
                        f"Data: {res.data}",
                    ]
                    + ["Routes:"]
                    + ["\t" + str(rule) for rule in current_app.url_map.iter_rules()]
                )
            )

    @staticmethod
    def _df_from_bytes(
        bytes_: bytes,
        file_type: TT.Literal[".csv", ".xlsx", ".xls"],
        header: T.Optional[int] = None,
        index_col: T.Optional[int] = None,
    ) -> pd.DataFrame:
        pass
        file_ = io.BytesIO(bytes_)
        if file_type == ".csv":
            df = pd.read_csv(file_, header=header, dtype=object)
        else:
            df = pd.read_excel(file_, header=header, dtype=object)
            df = df.astype(str)
        if (
            index_col is not None
        ):  # We don't pass index_col to read_csv/excel to avoid auto conversion(pandas being too heplful)
            df.set_index(df.columns[index_col], inplace=True)
        return df


class RQWorkerMixin(unittest.TestCase):
    """Allow spawning RQ workers with burst=True."""

    def setUp(self) -> None:
        super().setUp()
        settings.ensure_settings_initialized()
        self._redis_conn = Redis(host=Settings.REDIS_HOST, port=Settings.REDIS_PORT)

    def _burst_workers(self, queue_name: str) -> bool:
        queue = Queue(queue_name, connection=self._redis_conn)
        worker = Worker([queue], connection=self._redis_conn)
        return worker.work(burst=True)
