import logging
import typing as T

import redis
from rq import Queue  # type: ignore

from flask_app.modeling.classifier import ClassifierModel

logger = logging.getLogger(__name__)


class Scheduler(object):
    def __init__(self) -> None:
        self.my_redis = redis.Redis()
        self.classifier_training_queue = Queue(
            name="classifier_training", connection=self.my_redis
        )
        self.topic_modeling_queue = Queue(
            name="topic_modeling", connection=self.my_redis
        )

    def add_classifier_training(
        self,
        labels: T.List[str],
        model_path: str,
        data_dir: str,
        cache_dir: str,
        output_dir: str,
    ) -> None:
        pickle_data = {
            "labels": labels,
            "model_path": model_path,
            "data_dir": data_dir,
            "cache_dir": cache_dir,
            "output_dir": output_dir,
        }
        # self.my_redis.lpush(self.queue_name, pickle.dumps(pickle_data))
        logger.info(f"Enqueued classifier training with pickle_data={pickle_data}")
        self.classifier_training_queue.enqueue(do_classifier_training, pickle_data)

    def add_topic_model_training(self, input_dir: str, output_dir: str,) -> None:
        """

        Args:
            input_dir: Has to contain the directory with the csv/excel document.
            output_dir: The directory where the output files iwll go.
        """
        pickle_data = {"input_dir": input_dir, "output_dir": output_dir}
        # self.my_redis.lpush(self.queue_name, pickle.dumps(pickle_data))
        logger.info(f"Enqueued lda training with pickle_data={pickle_data}")
        self.topic_modeling_queue.enqueue(do_topic_modeling, pickle_data)


def do_classifier_training(pickle_data: T.Dict[str, T.Any]) -> None:
    classifier_model = ClassifierModel(
        pickle_data["labels"],
        pickle_data["model_path"],
        pickle_data["data_dir"],
        pickle_data["cache_dir"],
        pickle_data["output_dir"],
    )
    classifier_model.train()


def do_topic_modeling(pickle_data: T.Dict[str, T.Any]) -> None:
    pass
