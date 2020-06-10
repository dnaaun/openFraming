import enum
import typing as T

import peewee as pw  # type: ignore

DATABASE = "sqlite.db"
DEBUG = True
SECRET_KEY = "1kw=nxaf5ohgs@c#r5e6(o($kpdvp43zdtsdq=h+d-vxcsz(uj"

database = pw.SqliteDatabase(DATABASE)

# From: https://github.com/coleifer/peewee/issues/630
class EnumField(pw.CharField):
    """
    This class enable a Enum like field for Peewee
    """

    def __init__(
        self, enum_class: T.Type[enum.Enum], *args: T.Any, **kwargs: T.Any
    ) -> None:
        self._enum_class = enum_class
        super(pw.CharField, self).__init__(*args, **kwargs)

    def db_value(self, value: T.Any) -> str:
        return value.name  # type: ignore

    def python_value(self, value: T.Any) -> enum.Enum:
        return self._enum_class(value)


class BaseModel(pw.Model):
    class Meta:
        database = database


class _ProgressEnum(enum.Enum):
    TRAINING = "TRAINING"
    RUNNING_INFERENCE = "RUNNING_INFERENCE"
    DONE = "DONE"


class Classifier(BaseModel):
    category_names = pw.TextField()
    progress = EnumField(enum_class=_ProgressEnum)


class ExampleSet(BaseModel):
    # TODO: Add a SQL constraint to make sure
    # all examples' labels are one of from the list of
    # possible of possible labels for set
    pass


class Example(BaseModel):
    text = pw.TextField()
    label = pw.TextField()
