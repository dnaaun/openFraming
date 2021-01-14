"""Peewee database ORM."""
import enum
from flask_app.settings import settings
import functools
import typing as T

import peewee as pw


database_proxy = pw.DatabaseProxy()  # Create a proxy for our db.


SubClass = T.TypeVar("SubClass", bound="BaseModel")


class BaseModel(pw.Model):
    class Meta:
        database = database_proxy

    def refresh(self: SubClass) -> SubClass:
        # https://stackoverflow.com/a/32156865/13664712
        return type(self).get(self._pk_expr())  # type: ignore


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


if T.TYPE_CHECKING:
    ListFieldBase = pw.Field[T.List[str]]
else:
    ListFieldBase = pw.TextField

class ListField(ListFieldBase):
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

    def db_value(self, value: T.Optional[T.Any]) -> T.Optional[str]:
        """Validate and convert to string."""
        # Allow a None for an empty list
        if value is None:
            return None
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


class ClassifierMetrics(BaseModel):
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



class TopicModelMetrics(BaseModel):
    umass_coherence = pw.FloatField()


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
    training_or_inference_completed = pw.BooleanField(default=False)
    error_encountered = pw.BooleanField(default=False) 
    metrics = pw.ForeignKeyField(ClassifierMetrics, null=True)


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

    @classmethod
    def create(  # type: ignore[override]
        cls, name: str, category_names: T.List[str], notify_at_email: str
    ) -> "Classifier":
        """Mypy can't type check creation wihtout this."""
        return super(Classifier, cls).create(
            name=name, category_names=category_names, notify_at_email=notify_at_email
        )

    classifier_id = pw.AutoField(primary_key=True) 
    name = pw.TextField() 
    category_names = ListField() 
    trained_by_openFraming = pw.BooleanField(default=False) 
    train_set = pw.ForeignKeyField(LabeledSet, null=True) 
    notify_at_email = pw.TextField() 
    dev_set = pw.ForeignKeyField(LabeledSet, null=True) 

    test_sets: T.Iterable["TestSet"]  # provided by backref on TestSet


class TestSet(BaseModel):
    """This will be a prediction set.

    Attributes:
        id: set id.
        classifier: The classifier this set is intended for.
        name: User given name of the set.
        inference_completed: Whether the training or the inference has
            completed this set.
        error_encountered: Whether error was encontered during inference.
    """

    @classmethod
    def create(cls, name: str, classifier: Classifier, notify_at_email: str) -> "TestSet":  # type: ignore[override]
        return super(TestSet, cls).create(
            name=name, classifier=classifier, notify_at_email=notify_at_email
        )

    # TODO: The primary key here should be composite of classifier and id field.
    # Right now, we have checks in app.py to make sure a certain classifier id/test set
    # id combo exists, but that's not good design at all.
    id_: int = pw.AutoField(primary_key=True)  # type: ignore
    classifier: Classifier = pw.ForeignKeyField(Classifier, backref="test_sets")  # type: ignore

    name: str = pw.CharField()  # type: ignore
    notify_at_email: str = pw.TextField()  # type: ignore
    inference_began: bool = pw.BooleanField(default=False)  # type: ignore
    error_encountered: bool = pw.BooleanField(default=False)  # type: ignore
    inference_completed: bool = pw.BooleanField(default=False)  # type: ignore


class LDASet(BaseModel):
    id_: int = pw.AutoField(primary_key=True)  # type: ignore
    error_encountered: bool = pw.BooleanField(default=False)  # type: ignore
    lda_completed: bool = pw.BooleanField(default=False)  # type: ignore
    metrics: T.Optional[TopicModelMetrics] = pw.ForeignKeyField(TopicModelMetrics, null=True)  # type: ignore


class TopicModel(BaseModel):
    """."""

    @classmethod
    def create(  # type: ignore[override]
        cls, name: str, num_topics: int, notify_at_email: str, topic_names: T.List[str]
    ) -> "TopicModel":
        assert len(topic_names) == num_topics
        return super(TopicModel, cls).create(
            name=name,
            num_topics=num_topics,
            notify_at_email=notify_at_email,
            topic_names=topic_names,
        )

    id_ = pw.AutoField()
    name = pw.CharField()
    num_topics = pw.IntegerField()
    topic_names = ListField(null=True)
    lda_set = pw.ForeignKeyField(LDASet, null=True)
    notify_at_email = pw.TextField()

    # NOTE: The below is ONLY a type annotation.
    # The actual attribute is made available using "backreferences" in peewee
    semi_supervised_sets: T.Type["SemiSupervisedSet"]

    @property
    # https://github.com/coleifer/peewee/issues/1667#issuecomment-405095432
    def semi_supervised_set(self) -> "SemiSupervisedSet":
        return self.semi_supervised_sets.get()  # type: ignore


class SemiSupervisedSet(BaseModel):
    topic_model = pw.ForeignKeyField(TopicModel, backref="semi_supervised_sets")
    labeled_set = pw.ForeignKeyField(LabeledSet)
    clustering_completed = pw.BooleanField()


F = T.TypeVar("F", bound=T.Callable[..., T.Any])


def needs_database_init(func: F) -> F:
    """A decorator for connecting to the database first. When doing queued jobs, 
       we're in a different process(in the OS sense), so there's no database connection yet. 
    """

    # This functools.wraps is SUPER IMPORTANT because pickling the decorated function
    # fails without it, which is necessary for RQ.
    @functools.wraps(func)
    def wrapper(*args: T.Any, **kwargs: T.Any) -> T.Any:
        database = pw.SqliteDatabase(str(settings.DATABASE_FILE))
        database_proxy.initialize(database)
        return func(*args, **kwargs)

    return T.cast(F, wrapper)


MODELS: T.Tuple[T.Type[pw.Model], ...] = (
    ClassifierMetrics,
    LabeledSet,
    Classifier,
    TestSet,
    LDASet,
    TopicModel,
    SemiSupervisedSet,
    TopicModelMetrics,
)
