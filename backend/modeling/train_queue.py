from concurrent.futures import ThreadPoolExecutor
import pickle

import redis

from classifier import ClassifierModel


TRAINING_QUEUE_NAME = 'TRAIN'
PROCESSES = 2

class ModelScheduler(object):
	def __init__(self):
		self.my_redis = redis.Redis()
		self.queue_name = TRAINING_QUEUE_NAME
	def add_training_process(self, labels, model_path, data_dir):
		pickle_data = {'labels': labels, 'model_path': model_path, 'data_dir': data_dir}
		self.my_redis.lpush(self.queue_name, pickle.dumps(pickle_data))


def do_train(pickle_data):
	classifier_model = ClassifierModel(pickle_data['labels'], pickle_data['model_path'], pickle_data['data_dir'], './')
	print('beginning training')
	classifier_model.train()
	print('training happened')


class ModelTrainer(object):
	def __init__(self):
		self.training_queue_name = TRAINING_QUEUE_NAME
		self.my_redis = redis.Redis()

	def do(self):
		with ThreadPoolExecutor(max_workers=4) as e:
			while True:
				if self.my_redis.llen(self.training_queue_name) > 0:
					_, pickled = self.my_redis.brpop(self.training_queue_name)
					pickle_data = pickle.loads(pickled)
					e.submit(do_train, pickle_data)


# m = ModelTrainer()
# m.do()