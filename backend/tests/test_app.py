import tempfile
import typing as T
from pathlib import Path

import peewee as pw
import pytest
from flask import Flask

import db
from app import create_app


@pytest.fixture
def app() -> T.Iterator[Flask]:
    """Return a Flask() instance that uses a clean database, and a project directory."""

    temp_proj_dir = Path(tempfile.gettempdir())
    test_db = pw.SqliteDatabase(":memory:")

    # Bind model to test db, since we have a complete list of all models, we do not need
    # to recursively bind dependencies.
    # http://docs.peewee-orm.com/en/latest/peewee/database.html?highlight=bind#testing-peewee-applications
    test_db.bind(db.MODELS, bind_refs=False, bind_backrefs=False)
    test_db.connect()
    test_db.create_tables(db.MODELS)

    create_app(project_data_dir=temp_proj_dir)

    # Tear down
    test_db.drop_tables(db.MODELS) 
    test_db.close()
    temp_proj_dir = 
