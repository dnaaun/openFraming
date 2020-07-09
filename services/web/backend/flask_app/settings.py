"""Settings and constants in a centralized place."""
import functools
import logging
import os
import typing as T
from pathlib import Path

import typing_extensions as TT

logger = logging.getLogger(__name__)


class SettingsFromOutside(TT.TypedDict):
    """These settings must be set using enviornmetn variables."""

    PROJECT_DATA_DIRECTORY: str
    TRANSFORMERS_CACHE_DIRECTORY: T.Optional[str]
    MALLET_BIN_DIRECTORY: str
    FLASK_ENV: TT.Literal["development", "production"]
    REDIS_HOST: str
    REDIS_PORT: int


class Settings:
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

    PROJECT_DATA_DIRECTORY: Path
    TRANSFORMERS_CACHE_DIRECTORY: Path
    DATABASE_FILE: Path
    MALLET_BIN_DIRECTORY: Path
    FLASK_ENV: TT.Literal["development", "production"]
    REDIS_HOST: str
    REDIS_PORT: int

    SUPPORTED_NON_CSV_FORMATS: T.Set[str] = {".xls", ".xlsx"}

    _initialized_already = False

    @classmethod
    def is_initialized_already(cls) -> bool:
        return cls._initialized_already

    @classmethod
    def initialize_from_env(cls) -> None:
        try:
            any_flask_env = os.environ["FLASK_ENV"]
            assert any_flask_env in ["production", "development"]
            flask_env: TT.Literal["production", "development"] = any_flask_env  # type: ignore[assignment]

            settings_dict = SettingsFromOutside(
                {
                    "PROJECT_DATA_DIRECTORY": os.environ["PROJECT_DATA_DIRECTORY"],
                    "TRANSFORMERS_CACHE_DIRECTORY": os.environ[
                        "TRANSFORMERS_CACHE_DIRECTORY"
                    ],
                    "MALLET_BIN_DIRECTORY": os.environ["MALLET_BIN_DIRECTORY"],
                    "FLASK_ENV": flask_env,
                    "REDIS_HOST": os.environ["REDIS_HOST"],
                    "REDIS_PORT": int(os.environ["REDIS_PORT"]),
                }
            )
            cls.initialize_from_dict(settings_dict)
        except KeyError as e:
            logger.critical("You did not define one or more environment variable(s).")
            raise e
        except BaseException as e:
            logger.critical("You did not set one or more environment *correctly*.")
            raise e

    @classmethod
    def initialize_from_dict(cls, settings_dict: SettingsFromOutside) -> None:
        if cls._initialized_already:
            raise RuntimeError("Settings already initialized.")
        cls.PROJECT_DATA_DIRECTORY = Path(settings_dict["PROJECT_DATA_DIRECTORY"])
        if settings_dict["TRANSFORMERS_CACHE_DIRECTORY"] not in [None, ""]:
            assert (
                settings_dict["TRANSFORMERS_CACHE_DIRECTORY"] is not None
            )  # Make mypy happy
            cls.TRANSFORMERS_CACHE_DIRECTORY = Path(
                settings_dict["TRANSFORMERS_CACHE_DIRECTORY"]
            )
        else:
            cls.TRANSFORMERS_CACHE_DIRECTORY = (
                Path(cls.PROJECT_DATA_DIRECTORY) / "transformers_cache"
            )
        cls.DATABASE_FILE = Path(cls.PROJECT_DATA_DIRECTORY) / "sqlite.db"
        cls.MALLET_BIN_DIRECTORY = Path(settings_dict["MALLET_BIN_DIRECTORY"])
        cls.FLASK_ENV = settings_dict["FLASK_ENV"]
        cls.REDIS_HOST = settings_dict["REDIS_HOST"]
        cls.REDIS_PORT = settings_dict["REDIS_PORT"]
        cls._initialized_already = True

    @classmethod
    def deinitialize(cls) -> None:
        """ONLY FOR UNIT TESTING. DO NOT USE OTHERWISE TO CAUSE LESS CONFUSION."""

        for attr in [
            "PROJECT_DATA_DIRECTORY",
            "TRANSFORMERS_CACHE_DIRECTORY",
            "DATABASE_FILE",
            "MALLET_BIN_DIRECTORY",
            "FLASK_ENV",
            "REDIS_HOST",
            "REDIS_PORT",
        ]:
            if hasattr(cls, attr):
                delattr(cls, attr)
        cls._initialized_already = False


F = T.TypeVar("F", bound=T.Callable[..., T.Any])


@T.overload
def needs_settings_init(*, from_env: bool = True) -> T.Callable[[F], F]:
    ...


@T.overload
def needs_settings_init(
    *, from_dict: T.Optional[SettingsFromOutside] = None
) -> T.Callable[[F], F]:
    ...


def needs_settings_init(
    *, from_env: bool = True, from_dict: T.Optional[SettingsFromOutside] = None,
) -> T.Callable[[F], F]:
    """A second-order decorator to make sure settings are initialized.

    Args:
        from_env: Initialize settings from environment.
        from_dict: A dictionary to iniitalize settings from. 
    """

    assert from_env ^ (from_dict is not None)

    def decorator(func: F) -> F:

        # This functools.wraps is SUPER IMPORTANT because pickling the decorated function
        # fails without it, which is necessary for RQ
        @functools.wraps(func)  # type: ignore[no-redef]
        def wrapper(*args: T.Any, **kwargs: T.Any) -> T.Any:
            if not Settings.is_initialized_already():
                if from_env:
                    Settings.initialize_from_env()
                else:
                    assert from_dict is not None
                    Settings.initialize_from_dict(from_dict)
            return func(*args, **kwargs)

        return T.cast(F, wrapper)

    return decorator
