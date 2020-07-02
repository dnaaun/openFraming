import logging
import typing as T

from fakeredis import FakeStrictRedis  # type: ignore
from redis import Redis
from rq import Queue  # type: ignore

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
        self.classifier_training_queue = Queue(
            name="classifier_training", connection=connection, is_async=is_async
        )
        self.topic_modeling_queue = Queue(
            name="topic_modeling", connection=connection, is_async=is_async
        )

    def add_classifier_training(
        self,
        labels: T.List[str],
        model_path: str,
        data_dir: str,
        cache_dir: str,
        output_dir: str,
    ) -> None:
        logger.info("Enqueued classifier training")
        self.classifier_training_queue.enqueue(
            do_classifier_training,
            labels=labels,
            model_path=model_path,
            data_dir=data_dir,
            cache_dir=cache_dir,
            output_dir=output_dir,
        )

    def add_topic_model_training(
        self,
        training_file: str,
        num_topics: int,
        fname_keywords: str,
        fname_topics_by_doc: str,
        iterations: int = 1000,
    ) -> None:
        logger.info("Enqueued lda training with pickle_data")
        self.topic_modeling_queue.enqueue(
            do_topic_modeling,
            training_file=training_file,
            num_topics=num_topics,
            fname_keywords=fname_keywords,
            fname_topics_by_doc=fname_topics_by_doc,
            iterations=iterations,
        )


def do_classifier_training(
    labels: T.List[str],
    model_path: str,
    data_dir: str,
    cache_dir: str,
    output_dir: str,
) -> None:
    classifier_model = ClassifierModel(
        labels, model_path, data_dir, cache_dir, output_dir,
    )
    classifier_model.train()


def do_topic_modeling(
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
