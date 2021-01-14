"""Settings and constants in a centralized place."""
from __future__ import annotations
import functools
import logging
import os
from typing import Dict, Optional, Set, Union
from pydantic import BaseSettings
from pathlib import Path

from pydantic.class_validators import validator


logger = logging.getLogger(__name__)


class SettingsFromOutside(BaseSettings):
    """These settings must be set using env variables. Pydantic handles that."""

    PROJECT_DATA_DIRECTORY: Path
    TRANSFORMERS_CACHE_DIRECTORY: Optional[Path]

    # The path to the SQLITE database to use.
    # We usually set it to PROJECT_DATA_DIRECTORY/sqlite.db
    DATABASE_FILE: Path

    # The directory where the mallet binary is located.
    # The Dockefile sets this appropriately after extracting the tar.gz
    MALLET_BIN_DIRECTORY: str
    REDIS_HOST: str
    REDIS_PORT: int
    SERVER_NAME: str
    SENDGRID_API_KEY: Optional[str] = None
    SENDGRID_FROM_EMAIL: Optional[str] = None

    @validator("SENDGRID_API_KEY")
    def both_or_none(cls, v, values: Dict[str, Union[None, Path]]) -> None:
        if (v is None) != (values["SENDGRID_FROM_EMAIL"] is None):
            raise ValueError("Provide either both or none of SENDGRID_API_KEY and SENDGRID_FROM_EMAIL")

settings = SettingsFromOutside()

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

# File format related
SUPPORTED_NON_CSV_FORMATS: Set[str] = {".xls", ".xlsx"}
DEFAULT_FILE_FORMAT = ".xlsx"
