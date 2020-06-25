import logging
import typing as T

import redis
from rq import Queue  # type: ignore

from flask_app.modeling.classifier import ClassifierModel

logger = logging.getLogger(__name__)


class ModelScheduler(object):
    def __init__(self) -> None:
        self.my_redis = redis.Redis()
        self.queue = Queue(connection=self.my_redis)

    def add_training_process(
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
        logger.info(f"Enqueued training with output_dir={output_dir}")
        self.queue.enqueue(do_train, pickle_data)


def do_train(pickle_data: T.Dict[str, T.Any]) -> None:
    classifier_model = ClassifierModel(
        pickle_data["labels"],
        pickle_data["model_path"],
        pickle_data["data_dir"],
        pickle_data["cache_dir"],
        pickle_data["output_dir"],
    )
    classifier_model.train()
