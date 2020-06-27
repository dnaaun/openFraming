"""Everything that is not dealing with HTTP and that doesn't belong in modeling/."""
import csv
import io
import typing as T
from functools import lru_cache
from pathlib import Path

from flask import current_app
from werkzeug.exceptions import BadRequest

LABELLED_CSV_CONTENT_COL = "example"
LABELLED_CSV_LABEL_COL = "category"
TRANSFORMERS_MODEL = "bert-base-uncased"
TEST_SET_SPLIT = 0.2

# mypy doesn't support recrsive types, so this is the best we can do
Json = T.Optional[T.Union[T.List[T.Any], T.Dict[str, T.Any], int, str, bool]]


class Files:
    """A class for defining where files will be stored."""

    @classmethod
    @lru_cache()
    def project_data_dir(cls, ensure_exists: bool = True) -> Path:
        """Dir where all project related files will be stored."""
        dir_: Path = current_app.config["PROJECT_DATA_DIR"]
        if ensure_exists:
            cls._create_dir_if_not_exists(dir_)
        return dir_

    @classmethod
    @lru_cache()
    def supervised_dir(cls, ensure_exists: bool = True) -> Path:
        """Dir for classifier weights, training and inference data."""
        dir_ = cls.project_data_dir() / "supervised"
        if ensure_exists:
            cls._create_dir_if_not_exists(dir_)
        return dir_

    @classmethod
    @lru_cache
    def unsupervised_dir(cls, ensure_exists: bool = True) -> Path:
        """Dir for LDA results, training and inference data."""
        dir_ = cls.project_data_dir() / "unsupervised"
        if ensure_exists:
            cls._create_dir_if_not_exists(dir_)
        return dir_

    @classmethod
    @lru_cache()
    def classifier_dir(cls, classifier_id: int, ensure_exists: bool = False) -> Path:
        """Dir for files related to one classifier."""
        dir_ = cls.supervised_dir() / f"classifier_{classifier_id}"
        if ensure_exists:
            cls._create_dir_if_not_exists(dir_)
        return dir_

    @classmethod
    @lru_cache()
    def classifier_train_set_file(cls, classifier_id: int) -> Path:
        """CSV training file for classifier."""
        return cls.classifier_dir(classifier_id) / "train.csv"

    @classmethod
    @lru_cache()
    def classifier_dev_set_file(cls, classifier_id: int) -> Path:
        """CSV training file for classifier."""
        return cls.classifier_dir(classifier_id) / "dev.csv"

    @classmethod
    @lru_cache()
    def classifier_output_dir(
        cls, classifier_id: int, ensure_exists: bool = True
    ) -> Path:
        """Trained model output dir"""
        dir_ = cls.classifier_dir(classifier_id) / "trained_model/"
        if ensure_exists:
            cls._create_dir_if_not_exists(dir_)
        return dir_

    @classmethod
    def _create_dir_if_not_exists(cls, path: Path) -> None:
        """Creates directory indicated by `path` if it doesn't exist.

        Does not create more than one directory (ie, nested directories).
        """
        if not path.exists():
            path.mkdir()

    @classmethod
    @lru_cache()
    def topic_model_dir(cls, id_: int, ensure_exists: bool = False) -> Path:
        dir_ = cls.unsupervised_dir() / f"topic_model_{id_}"
        if ensure_exists:
            cls._create_dir_if_not_exists(dir_)
        return dir_

    @classmethod
    @lru_cache()
    def topic_model_training_file(cls, id_: int) -> Path:
        return cls.topic_model_dir(id_) / "train.csv"

    @classmethod
    @lru_cache()
    def topic_model_keywords_file(cls, id_: int) -> Path:
        return cls.topic_model_dir(id_) / "keywords_per_topic.csv"

    @classmethod
    @lru_cache()
    def topic_model_probabilities_by_example_file(cls, id_: int) -> Path:
        return cls.topic_model_dir(id_) / "probabilities_by_example.csv"


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
            text_stream = io.TextIOWrapper(file_)
            table = list(csv.reader(text_stream))
            text_stream.close()
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
