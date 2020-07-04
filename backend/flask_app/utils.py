"""Everything that is not dealing with HTTP and that doesn't belong in modeling/."""
import csv
import io
import typing as T
from pathlib import Path

import typing_extensions as TT
from flask import current_app
from werkzeug.exceptions import BadRequest

# Used by  both classifiers and lda models
CONTENT_COL = "Example"
PREDICTED_LABEL_COL = "Predicted category"

# Used by lda model only
ID_COL = "Id"
STEMMED_CONTENT_COL = "Simplified text"
MOST_LIKELY_TOPIC_COL = "Most likely topic"
TOPIC_PROPORTIONS_ROW = "Proportions"

# Used by classiifiers only
LABEL_COL = "Category"

TRANSFORMERS_MODEL = "distilbert-base-uncased"
TEST_SET_SPLIT = 0.2
MINIMUM_LDA_EXAMPLES = 20
DEFAULT_NUM_KEYWORDS_TO_GENERATE = 20
MAX_NUM_EXAMPLES_PER_TOPIC_IN_PREIVEW = 10

# mypy doesn't support recrsive types, so this is the best we can do
Json = T.Optional[T.Union[T.List[T.Any], T.Dict[str, T.Any], int, str, bool]]

ClassifierMetrics = TT.TypedDict(
    "ClassifierMetrics",
    {
        "accuracy": float,
        "macro_f1_score": float,
        "macro_recall": float,
        "macro_precision": float,
    },
)


class Files:
    """A class for defining where files will be stored."""

    @classmethod
    def project_data_dir(cls, ensure_exists: bool = True) -> Path:
        """Dir where all project related files will be stored."""
        dir_: Path = current_app.config["PROJECT_DATA_DIR"]
        if ensure_exists:
            cls._create_dir_if_not_exists(dir_)
        return dir_

    @classmethod
    def supervised_dir(cls, ensure_exists: bool = True) -> Path:
        """Dir for classifier weights, training and inference data."""
        dir_ = cls.project_data_dir() / "supervised"
        if ensure_exists:
            cls._create_dir_if_not_exists(dir_)
        return dir_

    @classmethod
    def unsupervised_dir(cls, ensure_exists: bool = True) -> Path:
        """Dir for LDA results, training and inference data."""
        dir_ = cls.project_data_dir() / "unsupervised"
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
        """
        Below is the format of the file.
        Note that the number of columns depends on the number of topics.
        The number of rows depends on `utils.DEFAULT_NUM_KEYWORDS_TO_GENERATE`.

        |             | 0                  | ...                 | 9                   |
        | word_0      | show               | ...                 | gun                 |
        | word_1      | police             | ...                 | obama               |
        | word_2      | gun                | ...                 | school              |
        | word_3      | coming             | ...                 | big                 |
        | word_4      | shooting           | ...                 | european            |
        | word_5      | eric               | ...                 | woman               |
        | word_6      | pruitt             | ...                 | gay                 |
        | word_7      | boy                | ...                 | reform              |
        | word_8      | plan               | ...                 | call                |
        | word_9      | people             | ...                 | governor            |
        | word_10     | injured            | ...                 | haspel              |
        | word_11     | schneiderman       | ...                 | ohio                |
        | word_12     | sarah              | ...                 | honor               |
        | word_13     | film               | ...                 | attack              |
        | word_14     | cambridge          | ...                 | smith               |
        | word_15     | day                | ...                 | bill                |
        | word_16     | johnson            | ...                 | kill                |
        | word_17     | mother             | ...                 | boy                 |
        | word_18     | gop                | ...                 | girl                |
        | word_19     | black              | ...                 | mueller             |
        | proportions | 0.0864685341429144 | ...                 | 0.07952367977703813 |
        """
        return cls.topic_model_dir(id_) / "keywords.xlsx"

    @classmethod
    def topic_model_topics_by_doc_file(cls, id_: int) -> Path:
        """

        Here's the format of the file. Number of columns depends on number of topics. 
	Just as the 'proba_topic_0' column, there will be a column for every topic.
        Number of rows depends on number of examples the user uploaded.
| a  | b  | c                                                                                     | d                                                                                                         | e                   | o                 |
| -- | -- | ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------- | ----------------- |
| a  | id | text                                                                                  | simplified text                                                                                           | proba_topic_0       | most likely topic |
| 0  | 1  | Florida Officer Who Was Filmed Shoving A Kneeling Black Protester Has Been Charged    | ['florida', 'officer', 'filmed', 'shoving', 'kneeling', 'black', 'protester', 'charged']                  | 0.1206896551724138  | 0                 |
| 1  | 2  | Fox News Host Ed Henry Fired After Sexual Misconduct Investigation                    | ['fox', 'news', 'host', 'henry', 'fired', 'sexual', 'misconduct', 'investigation']                        | 0.103448275862069   | 0                 |
| 2  | 3  | Hong Kong Police Make First Arrests Under New Security Law Imposed By China           | ['hong', 'kong', 'police', 'make', 'first', 'arrest', 'new', 'security', 'law', 'imposed', 'china']       | 0.1016949152542373  | 1                 |
| 3  | 4  | As Democrats Unveil Ambitious Climate Goals, House Lawmakers Press For Green Stimulus | ['democrat', 'unveil', 'ambitious', 'climate', 'goal', 'house', 'lawmaker', 'press', 'green', 'stimulus'] | 0.1166666666666666  | 2                 |
| 4  | 5  | Citing Racial Bias, San Francisco Will End Mug Shots Release                          | ['citing', 'racial', 'bias', 'san', 'francisco', 'end', 'mug', 'shot', 'release']                         | 0.1016949152542373  | 2                 |
| 5  | 6  | ‘Your Chewing Sounds Like Nails On A Chalkboard’: What Life With Misophonia Is Like   | ['chewing', 'sound', 'like', 'nail', 'chalkboard', 'life', 'misophonia', 'like']                          | 0.125               | 0                 |
| 6  | 7  | Puerto Rico’s Troubled Utility Is A Goldmine For U.S. Contractors                     | ['puerto', 'rico', 'troubled', 'utility', 'goldmine', 'contractor']                                       | 0.08928571428571429 | 1                 |
| 7  | 8  | Schools Provide Stability For Refugees. COVID-19 Upended That.                        | ['school', 'provide', 'stability', 'refugee', 'covid', 'upended']                                         | 0.1071428571428571  | 5                 |
| 8  | 9  | Jada Pinkett Smith Denies Claim Will Smith Gave Blessing To Alleged Affair            | ['jada', 'pinkett', 'smith', 'denies', 'claim', 'smith', 'gave', 'blessing', 'alleged', 'affair']         | 0.08333333333333333 | 2                 |
| 9  | 10 | College Students Test Positive For Coronavirus, Continue Going To Parties Anyway      | ['college', 'student', 'test', 'positive', 'coronavirus', 'continue', 'going', 'party', 'anyway']         | 0.1052631578947368  | 4                 |
| 10 | 11 | A TikTok User Noticed A Creepy Thing In ‘Glee’ You Can’t Unsee                        | ['tiktok', 'user', 'noticed', 'creepy', 'thing', 'cant', 'unsee']                                         | 0.125               | 0                 |
| 11 | 12 | Prince Harry Speaks Out Against Institutional Racism: It Has ‘No Place’ In Society    | ['prince', 'harry', 'speaks', 'institutional', 'racism', 'place', 'society']                              | 0.1052631578947368  | 8                 |
| 12 | 13 | A Poet — Yes, A Poet — Makes History On ‘America’s Got Talent’                        | ['poet', 'yes', 'poet', 'make', 'history', 'got', 'talent']                                               | 0.09090909090909091 | 9                 |
| 13 | 14 | I Ate At A Restaurant In What Was Once COVID-19’s Deadliest County                    | ['ate', 'restaurant', 'covids', 'deadliest', 'county']                                                    | 0.1272727272727273  | 0                 |
| 14 | 15 | This Is What Racial Trauma Does To The Body And Brain                                 | ['racial', 'trauma', 'body', 'brain']                                                                     | 0.1111111111111111  | 4                 |
| 15 | 16 | How To Avoid Bad Credit As Protections In The CARES Act Expire                        | ['avoid', 'bad', 'credit', 'protection', 'care', 'act', 'expire']                                         | 0.1228070175438596  | 0                 |
| 16 | 17 | Here’s Proof We Need Better Mental Health Care For People Of Color                    | ['here', 'proof', 'need', 'better', 'mental', 'health', 'care', 'people', 'color']                        | 0.1071428571428571  | 1                 |
| 17 | 18 | “I hope that this is real,” Lauren Boebert said of the deep-state conspiracy theory.  | ['hope', 'real', 'lauren', 'boebert', 'said', 'deepstate', 'conspiracy', 'theory']                        | 0.1052631578947369  | 6                 |
| 18 | 19 | U.S. Buys Virtually All Of Coronavirus Drug Remdesivir In The World                   | ['buy', 'virtually', 'coronavirus', 'drug', 'remdesivir', 'world']                                        | 0.08928571428571429 | 6                 | 
	"""
        return cls.topic_model_dir(id_) / "topics_by_doc.xlsx"


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

            if table == []:
                raise BadRequest("An empty file was uploaded.")
            # strip blanks
            table = [[cell.strip() for cell in row] for row in table]
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
