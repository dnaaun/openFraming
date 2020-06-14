"""Peewee database ORM."""
import enum
import typing as T

import peewee as pw  # type: ignore

DATABASE = "sqlite.db"
DEBUG = True
SECRET_KEY = "1kw=nxaf5ohgs@c#r5e6(o($kpdvp43zdtsdq=h+d-vxcsz(uj"

database = pw.SqliteDatabase(DATABASE)


class BaseModel(pw.Model):
    """Defines metaclass with database connection."""

    class Meta:
        """meta class."""

        database = database


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


class ProgressEnum(str, enum.Enum):
    """Progress field for Classifier.

    The inheritance from str is to support json serialization.
    """

    NOT_TRAINED = "NOT_TRAINED"
    TRAINING = "TRAINING"
    RUNNING_INFERENCE = "RUNNING_INFERENCE"
    DONE = "DONE"


class Classifier(BaseModel):  # noqa: D101
    name = pw.TextField()
    category_names = pw.TextField()
    progress = EnumField(enum_class=ProgressEnum, default=ProgressEnum.NOT_TRAINED)


class ExampleSet(BaseModel):  # noqa: D101
    # TODO: Add a SQL constraint to make sure
    # all examples' labels are one of from the list of
    # possible of possible labels for set
    pass


class Example(BaseModel):  # noqa: D101
    text = pw.TextField()
    label = pw.TextField()


def _create_tables() -> None:
    with database:
        database.create_tables([Classifier])


if __name__ == "__main__":
    _create_tables()
