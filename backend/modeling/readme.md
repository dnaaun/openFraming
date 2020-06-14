Using the LDA Modeling Module:
- specify the directory where you keep your files (hopefully excel or csv or tsv)
- specify the column that contains the text you want to analyze
- make a Corpus object (more documentation available in the file/coming soon)
- give that Corpus object to the LDAModeler
- run the LDAModeler's topic modeling functions

Using the Classifier module:
```
labels = [1, 2, 99]
filepath_base = '../../../Downloads/bu_com_climate_change_tweets_2063_deduplicated_splits/2/'
c = ClassifierModel(labels, 'bert-base-uncased', filepath_base, './')
c.train()
print(c.eval_model())
```

To run the queueing:
- get Redis; start a redis instance w/ default settings (redis-server in terminal)
- in one process run:
```
from train_queue import ModelTrainer
m = ModelTrainer()
m.do()
```
this will be the queue that keeps track of processes and schedules them.  
It will run until the server is shut down.
- in another process run:
```
from train_queue import ModelScheduler
s = ModelScheduler()
s.add_training_process([1, 2, 99], 'bert-base-uncased', '../../../Downloads/bu_com_climate_change_tweets_2063_deduplicated_splits/2/')
```