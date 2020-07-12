import logging
import typing as T

import typing_extensions as TT
from redis import Redis
from rq import Queue  # type: ignore

from flask_app.settings import Settings

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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


class QueueManager(object):
    def __init__(self) -> None:
        connection = Redis(host=Settings.REDIS_HOST, port=Settings.REDIS_PORT)
        is_async = True
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
            "flask_app.modeling.tasks.do_classifier_related_task",
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

        logger.info("Enqueued classifier training.")
        self.classifiers_queue.enqueue(
            "flask_app.modeling.tasks.do_classifier_related_task",
            ClassifierPredictionTaskArgs(
                test_set_id=test_set_id,
                task_type="prediction",
                labels=labels,
                model_path=model_path,
                test_file=test_file,
                cache_dir=cache_dir,
                test_output_file=test_output_file,
            ),
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
        logger.info("Enqueued lda training with pickle_data.")

        self.topic_models_queue.enqueue(
            "flask_app.modeling.tasks.do_topic_model_related_task",
            TopicModelTrainingTaskArgs(
                topic_model_id=topic_model_id,
                training_file=training_file,
                num_topics=num_topics,
                fname_keywords=fname_keywords,
                fname_topics_by_doc=fname_topics_by_doc,
                iterations=iterations,
                mallet_bin_directory=mallet_bin_directory,
            ),
            job_timeout=-1,
        )
