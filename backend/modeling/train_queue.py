from concurrent.futures import ThreadPoolExecutor
import pickle

import redis
from rq import Queue

from classifier import ClassifierModel


class ModelScheduler(object):
	def __init__(self):
		self.my_redis = redis.Redis()
		self.queue = Queue(connection=self.my_redis)
	def add_training_process(self, labels, model_path, data_dir):
		pickle_data = {'labels': labels, 'model_path': model_path, 'data_dir': data_dir}
		# self.my_redis.lpush(self.queue_name, pickle.dumps(pickle_data))
		self.queue.enqueue(do_train, pickle_data)

def do_train(pickle_data):
	classifier_model = ClassifierModel(pickle_data['labels'], pickle_data['model_path'], pickle_data['data_dir'], './')
	classifier_model.train()