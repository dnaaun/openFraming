"""Everything that is not dealing with HTTP and that doesn't belong in modeling/."""
import csv
import io
import typing as T
from functools import lru_cache
from pathlib import Path

from flask import current_app
from werkzeug.exceptions import BadRequest


class Files:
    """A class for defining where files will be stored."""

    @classmethod
    @lru_cache
    def project_data_dir(cls) -> Path:
        """Dir where all project related files will be stored."""
        return current_app.config["PROJECT_DATA_DIR"]  # type: ignore

    @classmethod
    @lru_cache
    def supervised_dir(cls) -> Path:
        """Dir for classifier weights, training and inference data."""
        return cls.project_data_dir() / "supervised"

    @classmethod
    @lru_cache
    def unsupervised_dir(cls) -> Path:
        """Dir for LDA results, training and inference data."""
        return cls.project_data_dir() / "unsupervised"

    @classmethod
    @lru_cache
    def classifier_dir(cls, classifier_id: int) -> Path:
        """Dir for files related to one classifier."""
        return cls.supervised_dir() / f"classifier_{classifier_id}"

    @classmethod
    @lru_cache
    def classifier_train_set_file(cls, classifier_id: int) -> Path:
        """CSV training file for classifier."""
        return cls.classifier_dir(classifier_id) / "train.csv"

    @classmethod
    @lru_cache
    def classifier_test_set_file(cls, classifier_id: int) -> Path:
        """CSV training file for classifier."""
        return cls.classifier_dir(classifier_id) / "dev.csv"


class Validate:
    @classmethod
    def csv_and_get_table(cls, file_: io.BytesIO) -> T.List[T.List[str]]:
        """Check if file_ is a valid CSV file and returns the contents.

        Args:
            file_: A file object.

        Returns:
            table: A list of list of strings.

        Raises:
            BadRequest:
        """
        try:
            # TODO: Check if the file size is too large
            # TODO: Maybe don't read all the file into memory even if it's small enough?
            # Not sure though.
            table = list(csv.reader(io.TextIOWrapper(file_)))
            return table
        except Exception as e:
            current_app.logger.warning(f"Invalid CSV file: {e}")
            raise BadRequest("What you uploaded is not a valid CSV file.")

    @classmethod
    def table_has_headers(
        cls, table: T.List[T.List[str]], headers: T.List[str]
    ) -> None:
        """Check if the "table"'s first row matches the headers provided, case insensitively.

        Args:
            table: A list of lists of strings.
            headers: A list of strings.

        Raises:
            BadRequest:
        """
        if not [h.lower() for h in table[0]] == [h.lower() for h in headers]:
            raise BadRequest(
                f"table has headers {table[0]}, but needs to have headers" f"{headers}"
            )

    @classmethod
    def table_has_num_columns(
        cls, table: T.List[T.List[str]], num_columns: int
    ) -> bool:
        """Check if the "table"'s first row matches the headers provided, case insensitively.

        Args:
            table: A list of lists of strings.
            num_columns: The number of columns expected.
        Returns:
            is_valid:
        """
        return len(table[0]) == num_columns
