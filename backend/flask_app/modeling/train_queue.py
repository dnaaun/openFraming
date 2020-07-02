import logging
import typing as T

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


class Scheduler(object):
    def __init__(self, do_tasks_sychronously: bool) -> None:
        """"

        Args:
            do_tasks_sychronously: If true, all the jobs are done synchronously. Used for unit
                testing.
        """
        if not do_tasks_sychronously:
            connection = Redis()
            is_async = True
        else:
            connection = FakeStrictRedis()
            is_async = False
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
            task_type="training",
            num_train_epochs=num_train_epochs,
            classifier_id=classifier_id,
            labels=labels,
            model_path=model_path,
            train_file=train_file,
            dev_file=dev_file,
            cache_dir=cache_dir,
            output_dir=output_dir,
        )

    def add_classifier_prediction(
        self,
        classifier_id: int,
        labels: T.List[str],
        model_path: str,
        test_file: str,
        cache_dir: str,
        output_dir: str,
    ) -> None:
        logger.info("Enqueued classifier training")
        self.classifiers_queue.enqueue(
            do_classifier_related_task,
            classifier_id=classifier_id,
            task_type="prediction",
            labels=labels,
            model_path=model_path,
            test_file=test_file,
            cache_dir=cache_dir,
            output_dir=output_dir,
        )

    def add_topic_model_training(
        self,
        topic_model_id: int,
        training_file: str,
        num_topics: int,
        fname_keywords: str,
        fname_topics_by_doc: str,
        iterations: int = 1000,
    ) -> None:
        logger.info("Enqueued lda training with pickle_data")
        self.topic_models_queue.enqueue(
            do_topic_model_related_task,
            task_type="training",
            topic_model_id=topic_model_id,
            training_file=training_file,
            num_topics=num_topics,
            fname_keywords=fname_keywords,
            fname_topics_by_doc=fname_topics_by_doc,
            iterations=iterations,
        )


def do_classifier_related_task(
    task_type: T.Literal["prediction", "training"],
    *,
    classifier_id: int,
    labels: T.List[str],
    model_path: str,
    cache_dir: str,
    num_train_epochs: T.Optional[float] = None,
    train_file: T.Optional[str] = None,
    test_file: T.Optional[str] = None,
    dev_file: T.Optional[str] = None,
    output_dir: T.Optional[str] = None,
) -> None:
    if task_type == "prediction":
        assert train_file is None
        assert dev_file is None
        assert test_file is not None
        assert output_dir is None
        raise NotImplementedError()
    if task_type == "training":
        assert num_train_epochs is not None
        assert train_file is not None
        assert dev_file is not None
        assert test_file is None
        assert output_dir is not None
        classifier_model = ClassifierModel(
            labels=labels,
            num_train_epochs=num_train_epochs,
            model_path=model_path,
            train_file=train_file,
            dev_file=dev_file,
            cache_dir=cache_dir,
            output_dir=output_dir,
        )
        metrics = classifier_model.train_and_evaluate()

        clsf = db.Classifier.get(db.Classifier.classifier_id == classifier_id)
        assert clsf.train_set is not None
        assert clsf.dev_set is not None
        clsf.train_set.training_or_inference_completed = True
        clsf.dev_set.training_or_inference_completed = True
        clsf.dev_set.metrics = db.Metrics(**metrics)
        clsf.dev_set.metrics.save()
        clsf.dev_set.save()
        clsf.train_set.save()
    else:
        raise ValueError(f"invalid task type {task_type}")


def do_topic_model_related_task(
    task_type: TT.Literal["training"],
    *,
    topic_model_id: int,
    training_file: str,
    num_topics: int,
    fname_keywords: str,
    fname_topics_by_doc: str,
    iterations: int,
) -> None:
    corpus = Corpus(
        file_name=training_file,
        content_column_name=utils.CONTENT_COL,
        id_column_name=utils.ID_COL,
    )
    lda_modeler = LDAModeler(corpus, iterations=iterations)
    lda_modeler.model_topics_to_spreadsheet(
        num_topics=num_topics,
        fname_keywords=fname_keywords,
        fname_topics_by_doc=fname_topics_by_doc,
    )
