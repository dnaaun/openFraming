"""Peewee database ORM."""
from __future__ import annotations

import enum
import typing as T

import peewee as pw


DATABASE = pw.SqliteDatabase("sqlite.db")
"""The database connection."""


class BaseModel(pw.Model):
    """Defines metaclass with database connection."""

    class Meta:
        """meta class."""

        database = DATABASE


# From: https://github.com/coleifer/peewee/issues/630
class EnumField(pw.CharField):
    """An EnumField for Peewee."""

    def __init__(
        self, enum_class: T.Type[enum.Enum], *args: T.Any, **kwargs: T.Any
    ) -> None:
        """init.

        Args:
            enum_class:
            *args: Passed to pw.CharField.
            *kwargs: Passed to pw.CharField.
        """
        self._enum_class = enum_class
        super().__init__(*args, **kwargs)

    def db_value(self, value: enum.Enum) -> str:
        """Convert enum to str."""
        return value.name

    def python_value(self, value: T.Any) -> enum.Enum:
        """Convert str to enum."""
        return self._enum_class(value)


class ListField(pw.TextField):
    """A field to facilitate storing lists of strings as a textfield."""

    def __init__(self, sep: str = ",", *args: T.Any, **kwargs: T.Any) -> None:
        """init.

        Args:
            sep: What separator to use to separate fields.
            *args: Passed to pw.CharField.
            *kwargs: Passed to pw.CharField.
        """
        assert len(sep) == 1
        self._sep = sep
        super().__init__(*args, **kwargs)

    def db_value(self, value: T.Optional[T.Any]) -> str:
        """Validate and convert to string."""
        # Allow a None for an empty list
        if value is None:
            return ""
        elif not isinstance(value, list) or set(map(type, value)) != {str}:
            raise ValueError("ListField stores lists of strings.")

        if any(self._sep in item for item in value):
            raise ValueError(
                f"ListField has separator {self._sep}, so a list item"
                " cannot have this character."
            )
        return self._sep.join(value)

    def python_value(self, value: T.Optional[str]) -> T.Optional[T.List[str]]:
        """Convert str to list."""
        if value is not None:
            return value.split(self._sep)
        return None


class ProgressEnum(str, enum.Enum):
    """Progress field for Classifier.

    The inheritance from str is to support json serialization.
    """

    NOT_TRAINED = "NOT_TRAINED"
    TRAINING = "TRAINING"
    RUNNING_INFERENCE = "RUNNING_INFERENCE"
    DONE = "DONE"


class Metrics(BaseModel):
    """Metrics on a labeled set.

    Attributes:
        macro_f1_score:
        macro_precision:
        macro_recall:
        accuracy:
    """

    macro_f1_score = pw.FloatField()
    macro_precision = pw.FloatField()
    macro_recall = pw.FloatField()
    accuracy = pw.FloatField()


class LabeledSet(BaseModel):
    """This is either a train set, or a dev set.

    We don't need a "name" field for this because there will only be one train set
    and one dev set per classifier. For the same reason, we don't store a foreign key
    to the classifier here.

    Attributes:
        id: set id.
        training_or_inference_completed: Whether the training or the inference has
            completed this set.
        metrics: Metrics on set. Can be null in the case of a train set.
    """

    id_ = pw.AutoField(primary_key=True)
    training_or_inference_completed: bool = pw.BooleanField(default=False)  # type: ignore
    metrics: Metrics = pw.ForeignKeyField(Metrics, null=True)  # type: ignore


class Classifier(BaseModel):
    """.

    Attributes:
        name: Name of classiifer.
        category_names: Comma separated names of categories. Means category names can't
            have commas.
        trained_by_openFraming: Whether this is a classifier that openFraming provides,
            or a user trained.
        train_set: The train set for classififer.
        dev_set: The dev set for classififer.
    """

    classifier_id: int = pw.AutoField(primary_key=True)
    name = pw.TextField()
    category_names: T.List[str] = ListField()  # type: ignore
    trained_by_openFraming: bool = pw.BooleanField(default=False)  # type: ignore
    train_set: T.Optional[LabeledSet] = pw.ForeignKeyField(LabeledSet, null=True)  # type: ignore
    dev_set: T.Optional[LabeledSet] = pw.ForeignKeyField(LabeledSet, null=True)  # type: ignore


class PredictionSet(BaseModel):
    """This will be a prediction set.

    Attributes:
        id: set id.
        classifier: The classifier this set is intended for.
        name: User given name of the set.
        inference_completed: Whether the training or the inference has
            completed this set.
    """

    id_ = pw.AutoField(primary_key=True)
    name = pw.CharField()
    classifier = pw.ForeignKeyField(Classifier, backref="prediction_sets")
    inference_completed = pw.BooleanField()


class LDASet(BaseModel):
    id_ = pw.AutoField(primary_key=True)
    lda_completed: bool = pw.BooleanField(default=False)  # type: ignore


class TopicModel(BaseModel):
    """."""

    id_: int = pw.AutoField()
    name: str = pw.CharField()
    num_topics: int = pw.IntegerField()
    topic_names: T.List[str] = ListField(null=True)  # type: ignore
    lda_set: T.Optional[LDASet] = pw.ForeignKeyField(LDASet, null=True)  # type: ignore

    # NOTE: The below is ONLY a type annotation.
    # The actual attribute is made available using "backreferences" in peewee
    semi_supervised_sets: T.Type[SemiSupervisedSet]

    @property
    # https://github.com/coleifer/peewee/issues/1667#issuecomment-405095432
    def semi_supervised_set(self) -> SemiSupervisedSet:
        return self.semi_supervised_sets.get()  # type: ignore


class SemiSupervisedSet(BaseModel):
    topic_model: TopicModel = pw.ForeignKeyField(TopicModel, backref="semi_supervised_sets")  # type: ignore
    labeled_set: LabeledSet = pw.ForeignKeyField(LabeledSet)  # type: ignore
    clustering_completed: bool = pw.BooleanField()  # type: ignore


MODELS = BaseModel.__subclasses__()


def _create_tables(database: pw.Database = DATABASE) -> None:
    """Create the tables in the database."""
    with database:
        database.create_tables(MODELS)


if __name__ == "__main__":
    _create_tables()
