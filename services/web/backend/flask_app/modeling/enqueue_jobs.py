import logging
import typing as T

import typing_extensions as TT
from redis import Redis
from rq import Queue  # type: ignore

from flask_app import db
from flask_app.modeling.classifier import ClassifierModel
from flask_app.modeling.lda import Corpus
from flask_app.modeling.lda import LDAModeler
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


@db.needs_database_init
def do_classifier_related_task(
    task_args: T.Union[ClassifierTrainingTaskArgs, ClassifierPredictionTaskArgs],
) -> None:
    if task_args["task_type"] == "prediction":
        test_set = db.TestSet.get(db.TestSet.id_ == task_args["test_set_id"])
        assert test_set.inference_began
        assert not test_set.inference_completed

        try:
            classifier_model = ClassifierModel(
                labels=task_args["labels"],
                model_path=task_args["model_path"],
                cache_dir=task_args["cache_dir"],
            )

            classifier_model.predict_and_save_predictions(
                test_set_path=task_args["test_file"],
                content_column=Settings.CONTENT_COL,
                predicted_column=Settings.PREDICTED_LABEL_COL,
                output_file_path=task_args["test_output_file"],
            )
        except BaseException as e:
            logger.critical(f"Error while doing prediction task: {e}")
            test_set.error_encountered = True
        else:
            test_set.inference_completed = True
        finally:
            test_set.save()

    elif task_args["task_type"] == "training":
        assert task_args["task_type"] == "training"
        clsf = db.Classifier.get(
            db.Classifier.classifier_id == task_args["classifier_id"]
        )
        assert clsf.train_set is not None
        assert clsf.dev_set is not None

        try:
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
        except BaseException as e:
            logger.critical(f"Error while doing classifier training task: {e}")
            clsf.train_set.error_encountered = True
            clsf.dev_set.error_encountered = True
        else:
            clsf.train_set.training_or_inference_completed = True
            clsf.dev_set.training_or_inference_completed = True
            clsf.dev_set.metrics = db.Metrics(**metrics)
            clsf.dev_set.metrics.save()
        finally:
            clsf.dev_set.save()
            clsf.train_set.save()
            clsf.save()


@db.needs_database_init
def do_topic_model_related_task(task_args: TopicModelTrainingTaskArgs) -> None:
    topic_mdl = db.TopicModel.get(db.TopicModel.id_ == task_args["topic_model_id"])
    assert topic_mdl.lda_set is not None
    try:
        corpus = Corpus(
            file_name=task_args["training_file"],
            content_column_name=Settings.CONTENT_COL,
            id_column_name=Settings.ID_COL,
        )
        lda_modeler = LDAModeler(
            corpus,
            iterations=task_args["iterations"],
            mallet_bin_directory=task_args["mallet_bin_directory"],
        )
    except BaseException as e:
        logger.critical(f"Error while doing lda training task: {e}")
        topic_mdl.lda_set.error_encountered = True
    else:
        lda_modeler.model_topics_to_spreadsheet(
            num_topics=task_args["num_topics"],
            fname_keywords=task_args["fname_keywords"],
            fname_topics_by_doc=task_args["fname_topics_by_doc"],
        )
        topic_mdl.lda_set.lda_completed = True
    finally:
        topic_mdl.lda_set.save()


class Scheduler(object):
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
            job_timeout=-1,
        )
