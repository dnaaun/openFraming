from __future__ import annotations

import csv
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

import peewee as pw
from flask import Response

from flask_app import db
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
        self._temp_proj_dir = Path(tempfile.mkdtemp(prefix="project_data_"))
        self._test_db = pw.SqliteDatabase(":memory:")
        self._app = create_app(
            project_data_dir=self._temp_proj_dir, do_tasks_sychronously=True
        )
        self._app.config["TESTING"] = True

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
