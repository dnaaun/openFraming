# Using the LDA Modeling Module:
- Your files should be .xlsx, .xls, .csv, or .tsv; they should all be the same type and live in the same directory.
- The code, by design, gives the user a lot of flexibility.  We'll elide that for now (MVP) and offer more of that later as time permits.
- The two objects we use in `lda.py` are the `Corpus` and the `LDAModeler`.  
- A `Corpus` object parses and holds text data; the LDAModeler runs LDA analysis on a Corpus object.
- You'll want to specify MALLET_PATH in lda.py. 

## Using the Corpus object:
You'll want to specify a few variables, and I've outlined the full domain of things that can be specified here:
- _dir_name_: a string. the path to the directory that holds the files you want to run LDA on.
- _content_column_name:_ a string. name of the column that holds the content you want to analyze.
- _language_: a string. default value is `'english'`. the language your text is written in.
- _min_word_length_: an int. minimum length of words that you want in the analysis. default value is 2.
- _extra_punctuation_: a set. default value is the empty set. any non-standard punctuation marks you would like removed from the analysis.
- _extra_stopwords_: a list. default value is an empty list. any words (no spaces) that are frequently seen in the text and non-informative go here.
- _phrases_to_join_: a list. default value is an empty list. any phrases (containing spaces) that you want recognized as a single entity (i.e. "Keanu Reeves" --> "keanu_reeves" in the final analysis)
- _phrases_to_remove_: a list. default value is an empty list. any phrases (containing spaces) that you want to remove from the analysis. An example might be "Subscribe to our mailing list!" - maybe you care about "mailing" and "subscribe", but you don't care about that whole phrase. 
- _dont_stem_: default value is the empty set. words you don't want to stem (convert to their root form) (e.g. fishy --> fish, potatoes --> potato, descended --> descend). An example of a word you might not want to stem is "Sanders" (as in Bernie Sanders).
- _header_: default value is False. Indicates whether the files you're loading have a header row (as pandas will interpret it)
- _processing_to_do_: indicates which text preprocessing steps to do or *avoid*. default value is the empty dict. Does all the steps listed below by default; if you want to remove a step, add it to the dict with the value False. Full list of preprocessing steps:
	- 'remove_phrases' (remove useless phrases)
	- 'join_phrases' (join together phrases with spaces into single entities)
	- 'remove_punctuation_and_digits' 
	- 'remove_stopwords' (remove commonly seen words that don't add meaning, like "and" or "the")
	- 'lemmatize_content' (stem content to root words, so "jumped" --> "jump" and "stairs" --> "stair")
	- 'remove_short_words' (remove words shorter than a user-supplied minimum length)

Example instantiation:
```
phrases_to_join = ['white house', 'social distancing']
phrases_to_remove = ['join our mailing list', 'click to subscribe']
dir_docs = '/disk2/test_docs/'
my_corpus = Corpus(dir_docs, 'Unnamed: 3', phrases_to_remove=phrases_to_remove, phrases_to_join=phrases_to_join, header=True)
```
## Using the LDAModeler:
Just give it a Corpus object: `modeler = LDAModeler(my_corpus)` - this instantiates the model. Once you've instantiated the model, you can call `model_topics(num_topics, num_keywords)`, which will give you a tuple of (topic_keywords, topic_proportions, topics_by_doc).  The default value for num_topics is 10, and the default value for num_keywords is 20. topic_keywords is the top num_keywords keywords; topic_proportions tells us how much each proportion makes up the corpus; and topics_by_doc is a list of topic probabilities by document (a matrix that is (len(documents) x num_topics)). 

Calling `model_topics_to_spreadsheet` requires a bit more information; you'll need to specify:
- num_topics (as discussed above - number of LDA topics - you can fiddle with this if needed). default value is 10.
- num_keywords (as discussed above - number of keywords you want to see for each LDA topic) . default value is 20.
- fname_keywords (the filename you'd like to dump your keywords to). default value is topic_keywords_and_proportions.xlsx', 
- name_topics_by_doc (the filename you'd like to dump your document topic probabilities to). default value is topic_probabilities_by_document.xlsx',
- extra_df_columns_wanted (the list of column names besides the content column that you'd like included in the document topic probabilities spreadsheet)
After you run it, you can use the output spreadsheets to learn more about the topics the LDA algorithm discovered.

# Using the Automatic Dataset Generator:
After you've run your LDA model, before discarding the object, collect a random sample from it (using the random_sample method in the Corpus object). Also be sure to download the `topic_probabilities_by_document.xlsx` spreadsheet. Label that random sample of data in the `EXPERT_LABEL` column. Then initialize the `SemiSupervisedDatasetCreator` object by giving it the filenames of the labeled sample and the topic probabilities by document:
```
s = SemiSupervisedDatasetCreator(
    "./topic_probabilities_by_document.xlsx",
    "../../../../Downloads/samples.xlsx",
    [1, 2, 99],
)
```
To get a labeled dataframe, run `s.get_labeled_df()`. This will return a dataframe with created labels. If you would like to have your labeled dataframe saved as a spreadsheet (ready to be uploaded to the Classifier module), run `s.get_labeled_df_to_spreadsheet(fname)`. 

# Using the Classifier module:
You'll need a list of labels (all the valid labels used in your training & eval sets). The set of labels in the training set and the set of labels in the eval set should be the same. You'll also need a path to the train & eval sets, which should be in .csv, .tsv, .xlsx, or .xls format. We're assuming they're named train.[extension] and eval.[extension], respectively. We'll also be using the `bert-base-uncased` model; although it's possible to use other models we should stick to this one for the MVP. Our output directory (where logs and model checkpoints will be saved to) is also indicated here.

Here's some example usage:
```
labels = [1, 2, 99]
filepath_base = '../../../Downloads/bu_com_climate_change_tweets_2063_deduplicated_splits/2/'
output_dir = './'
c = ClassifierModel(labels, 'bert-base-uncased', filepath_base, output_dir)
c.train() # train the model on the training set, using the eval set as validation
print(c.eval_model()) # evaluate the model's performance on the eval set
```

# Queueing Tasks:
We make use of Redis and rq to keep track of & run training tasks.
To run the queueing:
- make sure you have Redis installed
- start a redis instance w/ default settings (`redis-server` in the terminal)
- install rq: `pip3 install rq`
- in another terminal run `rq worker`. This will be the worker that keeps track of processes and schedules them. It will run until the Redis server is shut down.
- in another process run:
```
from train_queue import ModelScheduler
s = ModelScheduler()
labels = [1, 2, 99]
model_name = 'bert-base-uncased'
data_path = '../../../Downloads/bu_com_climate_change_tweets_2063_deduplicated_splits/2/'
s.add_training_process(labels, model_name, data_path) # adds a training task to the queue. 
```
This creates a ModelScheduler instance and adds it to the queue. We can watch the rq instance to see it run. 
