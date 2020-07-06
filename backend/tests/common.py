from __future__ import annotations

import csv
import functools
import io
import logging
import pdb
import shutil
import sys
import tempfile
import traceback
import typing as T
import unittest
from pathlib import Path

from flask import Response

from flask_app import db
from flask_app import utils
from flask_app.app import create_app

F = T.TypeVar("F", bound=T.Callable[[T.Any], T.Any])


TESTING_FILES_DIR = Path(__file__).parent / "testing_files"


def make_csv_file(table: T.List[T.List[str]]) -> io.BytesIO:
    text_io = io.StringIO()
    writer = csv.writer(text_io)
    writer.writerows(table)
    text_io.seek(0)
    csv_file = io.BytesIO(text_io.read().encode())
    return csv_file


def debug_on(*exceptions: T.Type[Exception]) -> T.Callable[[F], F]:
    # From stackoverflow.
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


class AppMixin(unittest.TestCase):
    def setUp(self) -> None:
        # NOTE: IF you remove the following, make sure to remove the shutil.rmtree call
        # in tearDown() as well.
        utils.PROJECT_DATA_DIRECTORY = Path(tempfile.mkdtemp(prefix="project_data_"))
        utils.DATABASE_FILE = (
            Path(tempfile.mkdtemp(prefix="project_data_")) / "sqlite.db"
        )
        self._app = create_app(do_tasks_synchronously=True, logging_level=logging.DEBUG)

        self._app.config["TESTING"] = True
        self._app.config["DEBUG"] = True

    def tearDown(self) -> None:
        db.database_proxy.drop_tables(db.MODELS)
        db.database_proxy.close()
        shutil.rmtree(utils.PROJECT_DATA_DIRECTORY)

    def _assert_response_success(
        self: AppMixin, res: Response, url: T.Optional[str] = None
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
                    + ["\t" + str(rule) for rule in self._app.url_map.iter_rules()]
                )
            )
