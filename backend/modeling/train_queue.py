import typing as T

import redis
from classifier import ClassifierModel
from rq import Queue  # type: ignore


class ModelScheduler(object):
    def __init__(self) -> None:
        self.my_redis = redis.Redis()
        self.queue = Queue(connection=self.my_redis)

    def add_training_process(
        self, labels: T.List[str], model_path: str, data_dir: str
    ) -> None:
        pickle_data = {"labels": labels, "model_path": model_path, "data_dir": data_dir}
        # self.my_redis.lpush(self.queue_name, pickle.dumps(pickle_data))
        self.queue.enqueue(do_train, pickle_data)


def do_train(pickle_data: T.Dict[str, T.Any]) -> None:
    classifier_model = ClassifierModel(
        # TODO: Make the cache_dir argument (which is a "./" right now) a variable
        pickle_data["labels"],
        pickle_data["model_path"],
        pickle_data["data_dir"],
        "./",
    )
    classifier_model.train()
