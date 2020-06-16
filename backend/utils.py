"""Everything that is not dealing with HTTP and that doesn't belong in modeling/."""
from functools import lru_cache
from pathlib import Path

from flask import current_app


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
    def classifier_dev_set_file(cls, classifier_id: int) -> Path:
        """CSV training file for classifier."""
        return cls.classifier_dir(classifier_id) / "dev.csv"
