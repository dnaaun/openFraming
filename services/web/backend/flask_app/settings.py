"""Settings and constants in a centralized place."""
import functools
import logging
import os
import typing as T
from pathlib import Path


logger = logging.getLogger(__name__)


class SettingsFromOutside(T.NamedTuple):
    """These settings must be set using enviornmetn variables."""

    PROJECT_DATA_DIRECTORY: str
    TRANSFORMERS_CACHE_DIRECTORY: T.Optional[str]
    MALLET_BIN_DIRECTORY: str
    REDIS_HOST: str
    REDIS_PORT: int
    SENDGRID_API_KEY: T.Optional[str]
    SENDGRID_FROM_EMAIL: T.Optional[str]
    SERVER_NAME: str


class Settings:
    ###### Very internal ####
    # Used by  both classifiers and lda models
    CONTENT_COL = "Example"
    PREDICTED_LABEL_COL = "Predicted category"

    # Used by lda model only
    ID_COL = "Id"
    STEMMED_CONTENT_COL = "Simplified text"
    MOST_LIKELY_TOPIC_COL = "Most likely topic"
    TOPIC_PROPORTIONS_ROW = "Proportions"
    DEFAULT_TOPIC_NAME_TEMPLATE = "Topic {}"
    PROBAB_OF_TOPIC_TEMPLATE = "Probability of topic: {}"

    # Used by classiifiers only
    LABEL_COL = "Category"

    TRANSFORMERS_MODEL = "distilbert-base-uncased"
    TEST_SET_SPLIT = 0.2
    MINIMUM_LDA_EXAMPLES = 20
    DEFAULT_NUM_KEYWORDS_TO_GENERATE = 20
    MAX_NUM_EXAMPLES_PER_TOPIC_IN_PREIVEW = 10
    ###### End of: very internal ####

    # File format related
    SUPPORTED_NON_CSV_FORMATS: T.Set[str] = {".xls", ".xlsx"}
    DEFAULT_FILE_FORMAT = ".xlsx"

    # These depend on envionment varialbes
    PROJECT_DATA_DIRECTORY: Path
    TRANSFORMERS_CACHE_DIRECTORY: Path
    DATABASE_FILE: Path
    MALLET_BIN_DIRECTORY: Path
    REDIS_HOST: str
    REDIS_PORT: int
    SENDGRID_API_KEY: T.Optional[str]
    SENDGRID_FROM_EMAIL: T.Optional[str]
    SERVER_NAME: str

    _initialized_already = False

    @classmethod
    def repr(cls) -> str:
        return str({name: val for name, val in vars(cls).items() if not callable(val)})

    @classmethod
    def is_initialized_already(cls) -> bool:
        return cls._initialized_already

    @classmethod
    def initialize_from_env(cls) -> None:
        try:
            print('here from env')
            settings_tup = SettingsFromOutside(
                PROJECT_DATA_DIRECTORY=os.environ["PROJECT_DATA_DIRECTORY"],
                TRANSFORMERS_CACHE_DIRECTORY=os.environ["TRANSFORMERS_CACHE_DIRECTORY"],
                MALLET_BIN_DIRECTORY=os.environ["MALLET_BIN_DIRECTORY"],
                REDIS_HOST=os.environ["REDIS_HOST"],
                REDIS_PORT=int(os.environ["REDIS_PORT"]),
                SENDGRID_API_KEY=os.environ.get("SENDGRID_API_KEY", None),
                SENDGRID_FROM_EMAIL=os.environ.get("SENDGRID_FROM_EMAIL", None),
                SERVER_NAME=os.environ["SERVER_NAME"],
            )
            if bool(settings_tup.SENDGRID_FROM_EMAIL) != bool(
                settings_tup.SENDGRID_FROM_EMAIL
            ):
                raise RuntimeError(
                    "Either both need to be set, or both need to be not set."
                )
            cls.initialize_from_tup(settings_tup)
        except KeyError as e:
            logger.critical("You did not define one or more environment variable(s).")
            raise e
        except BaseException as e:
            logger.critical("You did not set one or more environment *correctly*.")
            raise e

    @classmethod
    def initialize_from_tup(cls, settings_tup: SettingsFromOutside) -> None:
        if cls._initialized_already:
            raise RuntimeError("Settings already initialized.")
        cls.PROJECT_DATA_DIRECTORY = Path(settings_tup.PROJECT_DATA_DIRECTORY)
        if settings_tup.TRANSFORMERS_CACHE_DIRECTORY not in [None, ""]:
            assert (
                settings_tup.TRANSFORMERS_CACHE_DIRECTORY is not None
            )  # Make mypy happy
            cls.TRANSFORMERS_CACHE_DIRECTORY = Path(
                settings_tup.TRANSFORMERS_CACHE_DIRECTORY
            )
        else:
            cls.TRANSFORMERS_CACHE_DIRECTORY = (
                Path(cls.PROJECT_DATA_DIRECTORY) / "transformers_cache"
            )
        cls.DATABASE_FILE = Path(cls.PROJECT_DATA_DIRECTORY) / "sqlite.db"
        cls.MALLET_BIN_DIRECTORY = Path(settings_tup.MALLET_BIN_DIRECTORY)
        cls.REDIS_HOST = settings_tup.REDIS_HOST
        cls.REDIS_PORT = settings_tup.REDIS_PORT
        cls.SENDGRID_API_KEY = settings_tup.SENDGRID_API_KEY
        cls.SENDGRID_FROM_EMAIL = settings_tup.SENDGRID_FROM_EMAIL
        cls.SERVER_NAME = settings_tup.SERVER_NAME
        cls._initialized_already = True
        print('hereh')
        if cls.SENDGRID_API_KEY is None:
            logger.info(
                "Env variable SENDGRID_API_KEY was None, "
                "emails will not actually be sent, just printed to the console."
            )

    @classmethod
    def deinitialize(cls) -> None:
        """ONLY FOR UNIT TESTING. DO NOT USE OTHERWISE TO CAUSE LESS CONFUSION."""

        for attr in SettingsFromOutside._fields:
            if hasattr(cls, attr):
                delattr(cls, attr)
        cls._initialized_already = False


F = T.TypeVar("F", bound=T.Callable[..., T.Any])


# TODO: Ugly. Why have a decorator, *and* non decorator way of doing things?
def ensure_settings_initialized(
    from_tup: T.Optional[SettingsFromOutside] = None,
) -> None:
    """Ensures initalization, by either reading from env, or from dict."""
    if Settings.is_initialized_already():
        return
    if from_tup is None:
        Settings.initialize_from_env()
    else:
        assert from_tup is not None
        Settings.initialize_from_tup(from_tup)


def needs_settings_init(
    *, from_tup: T.Optional[SettingsFromOutside] = None,
) -> T.Callable[[F], F]:
    """A second-order decorator to make sure settings are initialized.

    Args:
        from_tup: A dictionary to iniitalize settings from. 
    """

    def decorator(func: F) -> F:

        # This functools.wraps is SUPER IMPORTANT because pickling the decorated function
        # fails without it, which is necessary for RQ
        @functools.wraps(func)  # type: ignore[no-redef]
        def wrapper(*args: T.Any, **kwargs: T.Any) -> T.Any:
            ensure_settings_initialized(from_tup=from_tup)
            return func(*args, **kwargs)

        return T.cast(F, wrapper)

    return decorator
