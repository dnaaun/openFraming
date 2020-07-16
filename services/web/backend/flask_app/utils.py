"""Everything that is not dealing with HTTP and that doesn't belong in modeling/."""
import csv
import hashlib
import io
import mimetypes
import typing as T
from pathlib import Path

import pandas as pd  # type: ignore
from flask import current_app
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import BadRequest

from flask_app.settings import Settings


# mypy doesn't support recrsive types, so this is the best we can do
Json = T.Optional[T.Union[T.List[T.Any], T.Dict[str, T.Any], int, str, bool]]


class Files:
    """A class for defining where files will be stored."""

    @classmethod
    def supervised_dir(cls, ensure_exists: bool = True) -> Path:
        """Dir for classifier weights, training and inference data."""
        dir_ = Settings.PROJECT_DATA_DIRECTORY / "supervised"
        if ensure_exists:
            cls._create_dir_if_not_exists(dir_)
        return dir_

    @classmethod
    def unsupervised_dir(cls, ensure_exists: bool = True) -> Path:
        """Dir for LDA results, training and inference data."""
        dir_ = Settings.PROJECT_DATA_DIRECTORY / "unsupervised"
        if ensure_exists:
            cls._create_dir_if_not_exists(dir_)
        return dir_

    @classmethod
    def classifier_dir(cls, classifier_id: int, ensure_exists: bool = False) -> Path:
        """Dir for files related to one classifier."""
        dir_ = cls.supervised_dir() / f"classifier_{classifier_id}"
        if ensure_exists:
            cls._create_dir_if_not_exists(dir_)
        return dir_

    @classmethod
    def classifier_train_set_file(cls, classifier_id: int) -> Path:
        """CSV training file for classifier."""
        return cls.classifier_dir(classifier_id) / "train.csv"

    @classmethod
    def classifier_dev_set_file(cls, classifier_id: int) -> Path:
        """CSV training file for classifier."""
        return cls.classifier_dir(classifier_id) / "dev.csv"

    @classmethod
    def classifier_output_dir(
        cls, classifier_id: int, ensure_exists: bool = True
    ) -> Path:
        """Trained model output dir"""
        dir_ = cls.classifier_dir(classifier_id) / "trained_model/"
        if ensure_exists:
            cls._create_dir_if_not_exists(dir_)
        return dir_

    @classmethod
    def classifier_test_set_dir(
        cls, classifier_id: int, test_set_id: int, ensure_exists: bool = True
    ) -> Path:
        """Files related to one prediction set will be stored here"""
        dir_ = cls.classifier_dir(classifier_id) / f"prediction_set_{test_set_id}"
        if ensure_exists:
            cls._create_dir_if_not_exists(dir_)
        return dir_

    @classmethod
    def classifier_test_set_file(cls, classifier_id: int, test_set_id: int) -> Path:
        """CSV test file for classifier."""
        return cls.classifier_test_set_dir(classifier_id, test_set_id) / "test.csv"

    @classmethod
    def classifier_test_set_predictions_file(
        cls, classifier_id: int, test_set_id: int
    ) -> Path:
        """CSV file for predictions on a test set."""
        return (
            cls.classifier_test_set_dir(classifier_id, test_set_id) / "predictions.csv"
        )

    @classmethod
    def _create_dir_if_not_exists(cls, path: Path) -> None:
        """Creates directory indicated by `path` if it doesn't exist.

        Does not create more than one directory (ie, nested directories).
        """
        if not path.exists():
            path.mkdir()

    @classmethod
    def topic_model_dir(cls, id_: int, ensure_exists: bool = False) -> Path:
        dir_ = cls.unsupervised_dir() / f"topic_model_{id_}"
        if ensure_exists:
            cls._create_dir_if_not_exists(dir_)
        return dir_

    @classmethod
    def topic_model_training_file(cls, id_: int) -> Path:
        return cls.topic_model_dir(id_) / "train.csv"

    @classmethod
    def topic_model_keywords_file(cls, id_: int) -> Path:
        return cls.topic_model_dir(id_) / "keywords.csv"

    @classmethod
    def topic_model_keywords_with_topic_names_file(
        cls, id_: int, topic_names: T.List[str]
    ) -> Path:
        """"

        This is where we store the "keywords" file that has been modified to contain
        the topic names provided by the user.
        """

        return (
            cls.topic_model_dir(id_)
            / f"keywords_with_topic_names_{cls._hash_list(topic_names)}.csv"
        )

    @classmethod
    def topic_model_topics_by_doc_with_topic_names_file(
        cls, id_: int, topic_names: T.List[str]
    ) -> Path:
        """" 
        """

        return (
            cls.topic_model_dir(id_)
            / f"topics_by_docs_with_topic_names_{cls._hash_list(topic_names)}.csv"
        )

    @classmethod
    def topic_model_topics_by_doc_file(cls, id_: int) -> Path:
        return cls.topic_model_dir(id_) / "topics_by_doc.csv"

    @staticmethod
    def _hash_list(ls: T.List[str]) -> str:
        """"Hash a list of items to get a fixed length representation"""
        assert all("," not in item for item in ls)
        hash_ = hashlib.sha1(",".join(ls).encode()).hexdigest()  # nosec
        return hash_


class Validate:
    @classmethod
    def spreadsheet_and_get_table(cls, file_: FileStorage) -> T.List[T.List[str]]:
        """Check if file_ is a valid csv,xls, xlsx, xlsm, xlsb, or odf
            file and returns the contents.

            This depends on whatever Flask says is the mimetype, which only checks
            HTTP headers and the file extension(doesn't inspect

        Args:
            file_: A file object.

        Returns:
            table: A list of list of strings.

        Raises:
            BadRequest:
        """

        # Something in the combination of the Python version used (3.8.3), and the fact
        # that it is containerized(run inside Docker) neccesitates this.
        mimetypes.add_type(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx"
        )

        file_type = mimetypes.guess_extension(file_.mimetype)
        if not file_type:
            raise BadRequest(
                "The uploaded file type could not be inferred."
                " Perhaps using a different browser might help."
            )

        if file_type in Settings.SUPPORTED_NON_CSV_FORMATS:
            try:
                df: pd.DataFrame = pd.read_excel(file_, na_filter=False, header=None)  # type: ignore
            except BaseException:
                raise BadRequest("The uploaded spreadsheet could not be parsed.")
            else:
                df = df.astype(str)
                table: T.List[T.List[str]] = df.to_numpy().tolist()  # type: ignore
        elif file_type in [".csv", ".txt"]:
            try:
                # TODO: Check if the file size is too large
                text_stream = io.TextIOWrapper(T.cast(io.BytesIO, file_))
                table = list(csv.reader(text_stream))
            except Exception as e:
                current_app.logger.info(f"Invalid CSV file: {e}")
                raise BadRequest("Uploaded text file is not in valid CSV format.")
            else:
                if table == []:
                    raise BadRequest("An empty file was uploaded.")
                # strip blanks
                table = [[cell.strip() for cell in row] for row in table]
                text_stream.close()
        else:
            raise BadRequest(
                f"File type {file_type} was not understood as a valid spreadhseet type,"
                "please upload one of the following file formats: "
                + ", ".join(Settings.SUPPORTED_NON_CSV_FORMATS | {".csv"})
            )

        return table

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
    def table_has_no_empty_cells(cls, table: T.List[T.List[str]]) -> None:
        """

        Does not strip off blanks.(That's done in Validate.csv_and_get_table

        Raises:
            BadRequest:
        """
        rows_with_empty_cells = []
        for row_num, row in enumerate(table, start=1):
            if any([cell == "" for cell in row]):
                rows_with_empty_cells.append(row)
        if rows_with_empty_cells:
            raise BadRequest(
                "The following row numbers have empty cells: "
                + ", ".join(map(str, rows_with_empty_cells))
            )

    @classmethod
    def table_has_num_columns(
        cls, table: T.List[T.List[str]], num_columns: int
    ) -> None:
        """Check if the "table"'s first row matches the headers provided, case insensitively.

        Args:
            table: A list of lists of strings.
            num_columns: The number of columns expected.
        Returns:
            is_valid:
        """
        if not len(table[0]) == num_columns:
            raise BadRequest(
                f"table must have {num_columns}"
                + ("column." if num_columns == 1 else "columns.")
            )
