import functools
import logging
import typing as T

import peewee as pw
import typing_extensions as TT
from fakeredis import FakeStrictRedis  # type: ignore
from redis import Redis
from rq import Queue  # type: ignore

from flask_app import db
from flask_app import utils
from flask_app.modeling.classifier import ClassifierModel
from flask_app.modeling.lda import Corpus
from flask_app.modeling.lda import LDAModeler

logger = logging.getLogger(__name__)


class ClassifierPredictionTaskArgs(TT.TypedDict):
    task_type: TT.Literal["prediction"]  # This redundancy is for mypy's sake
    test_set_id: int
    test_file: str
    labels: T.List[str]
    model_path: str
    cache_dir: str
    test_output_file: str


class ClassifierTrainingTaskArgs(TT.TypedDict):
    task_type: TT.Literal["training"]
    classifier_id: int
    labels: T.List[str]
    model_path: str
    cache_dir: str
    num_train_epochs: float
    train_file: str
    dev_file: str
    output_dir: str


class TopicModelTrainingTaskArgs(TT.TypedDict):
    # No need for task type, since the topic model queue has only one kind of task
    topic_model_id: int
    training_file: str
    num_topics: int
    fname_keywords: str
    fname_topics_by_doc: str
    iterations: int
    mallet_bin_directory: str


FuncT = T.TypeVar("FuncT", bound=T.Callable[..., T.Any])


def may_need_database_init(func: FuncT) -> T.Callable[..., T.Any]:
    """A decorator for connecting to the database first. When doing queued jobs, 
       we're in a different process, so there's no database connection yet. 

    Returns:
        func: A function that takes the keyword argument `init_database`, specifiying
        if the database initialization should be made before proceeding with the wrapped
        function.
    """
    wrapper: FuncT

    @functools.wraps(func)  # type: ignore[no-redef]
    def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def,no-redef]
        init_database = kwargs.pop("init_database")
        if init_database:
            database = pw.SqliteDatabase(str(utils.DATABASE_FILE))
            db.database_proxy.initialize(database)
        return func(*args, **kwargs)

    return wrapper


@may_need_database_init
def do_classifier_related_task(
    task_args: T.Union[ClassifierTrainingTaskArgs, ClassifierPredictionTaskArgs],
) -> None:
    if task_args["task_type"] == "prediction":
        test_set = db.TestSet.get(db.TestSet.id_ == task_args["test_set_id"])
        assert test_set.inference_began
        assert not test_set.inference_completed

        classifier_model = ClassifierModel(
            labels=task_args["labels"],
            model_path=task_args["model_path"],
            cache_dir=task_args["cache_dir"],
        )

        classifier_model.predict_and_save_predictions(
            test_set_path=task_args["test_file"],
            content_column=utils.CONTENT_COL,
            predicted_column=utils.PREDICTED_LABEL_COL,
            output_file_path=task_args["test_output_file"],
        )

        test_set.inference_completed = True
        test_set.save()

    elif task_args["task_type"] == "training":
        assert task_args["task_type"] == "training"
        clsf = db.Classifier.get(
            db.Classifier.classifier_id == task_args["classifier_id"]
        )
        assert clsf.train_set is not None
        assert clsf.dev_set is not None

        classifier_model = ClassifierModel(
            labels=task_args["labels"],
            num_train_epochs=task_args["num_train_epochs"],
            model_path=task_args["model_path"],
            train_file=task_args["train_file"],
            dev_file=task_args["dev_file"],
            cache_dir=task_args["cache_dir"],
            output_dir=task_args["output_dir"],
        )
        metrics = classifier_model.train_and_evaluate()

        clsf.train_set.training_or_inference_completed = True
        clsf.dev_set.training_or_inference_completed = True
        clsf.dev_set.metrics = db.Metrics(**metrics)
        clsf.dev_set.metrics.save()
        clsf.dev_set.save()
        clsf.train_set.save()


@may_need_database_init
def do_topic_model_related_task(task_args: TopicModelTrainingTaskArgs) -> None:
    corpus = Corpus(
        file_name=task_args["training_file"],
        content_column_name=utils.CONTENT_COL,
        id_column_name=utils.ID_COL,
    )
    lda_modeler = LDAModeler(
        corpus,
        iterations=task_args["iterations"],
        mallet_bin_directory=task_args["mallet_bin_directory"],
    )
    lda_modeler.model_topics_to_spreadsheet(
        num_topics=task_args["num_topics"],
        fname_keywords=task_args["fname_keywords"],
        fname_topics_by_doc=task_args["fname_topics_by_doc"],
    )


class Scheduler(object):
    def __init__(self, do_tasks_synchronously: bool) -> None:
        """"

        Args:
            do_tasks_synchronously: If true, all the jobs are done synchronously. Also, 
                no new database connection is created.

                Used for unit testing.
        """
        if not do_tasks_synchronously:
            connection = Redis()
            is_async = True
        else:
            connection = FakeStrictRedis()
            is_async = False
        self._do_tasks_synchronously = do_tasks_synchronously
        self.classifiers_queue = Queue(
            name="classifiers", connection=connection, is_async=is_async
        )
        self.topic_models_queue = Queue(
            name="topic_models", connection=connection, is_async=is_async
        )

    def add_classifier_training(
        self,
        classifier_id: int,
        labels: T.List[str],
        model_path: str,
        train_file: str,
        dev_file: str,
        cache_dir: str,
        output_dir: str,
        num_train_epochs: float = 3.0,
    ) -> None:
        logger.info("Enqueued classifier training")

        self.classifiers_queue.enqueue(
            do_classifier_related_task,
            ClassifierTrainingTaskArgs(
                task_type="training",
                classifier_id=classifier_id,
                num_train_epochs=num_train_epochs,
                labels=labels,
                model_path=model_path,
                train_file=train_file,
                dev_file=dev_file,
                cache_dir=cache_dir,
                output_dir=output_dir,
            ),
            init_database=not self._do_tasks_synchronously,  # This is for our decorator
            job_timeout=-1,  # This will be popped off by RQ
        )

    def add_classifier_prediction(
        self,
        test_set_id: int,
        labels: T.List[str],
        model_path: str,
        test_file: str,
        cache_dir: str,
        test_output_file: str,
    ) -> None:

        logger.info("Enqueued classifier training")
        self.classifiers_queue.enqueue(
            do_classifier_related_task,
            ClassifierPredictionTaskArgs(
                test_set_id=test_set_id,
                task_type="prediction",
                labels=labels,
                model_path=model_path,
                test_file=test_file,
                cache_dir=cache_dir,
                test_output_file=test_output_file,
            ),
            init_database=not self._do_tasks_synchronously,
            job_timeout=-1,
        )

    def add_topic_model_training(
        self,
        topic_model_id: int,
        training_file: str,
        num_topics: int,
        fname_keywords: str,
        fname_topics_by_doc: str,
        mallet_bin_directory: str,
        iterations: int = 1000,
    ) -> None:
        logger.info("Enqueued lda training with pickle_data")

        self.topic_models_queue.enqueue(
            do_topic_model_related_task,
            TopicModelTrainingTaskArgs(
                topic_model_id=topic_model_id,
                training_file=training_file,
                num_topics=num_topics,
                fname_keywords=fname_keywords,
                fname_topics_by_doc=fname_topics_by_doc,
                iterations=iterations,
                mallet_bin_directory=mallet_bin_directory,
            ),
            init_database=not self._do_tasks_synchronously,
            job_timeout=-1,
        )
