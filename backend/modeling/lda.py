import collections
import os
import numpy as np
import pandas as pd
import re
import string

from gensim import models, corpora
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


EXCEL_EXTENSIONS = {'xlsx', 'xls'}
CSV_EXTENSIONS = {'csv'}
TSV_EXTENSIONS = {'tsv'}

MALLET_PATH = '~/Downloads/mallet-2.0.8/bin/mallet'

class Corpus(object):
	"""
	"""
	def __init__(
		self, 
		dir_name: str, 
		content_column_name: str, 
		language='english', 
		min_word_length=2,
		extra_punctuation=set(),
		extra_stopwords=[],
		phrases_to_join=[], 
		phrases_to_remove=[], 
		dont_stem=set(),
		header=False,
		processing_to_do={}
	):
		dir_contents = os.listdir(dir_name)

		if len(dir_contents) < 1:
			raise ValueError('Directory {} is empty!'.format(dir_name))

		suffixes = set([fname.split('.')[-1] for fname in dir_contents])

		if suffixes.issubset(EXCEL_EXTENSIONS):
			doc_reader = pd.read_excel
		elif suffixes.issubset(CSV_EXTENSIONS):
			doc_reader = pd.read_csv
		elif suffixes.issubset(TSV_EXTENSIONS):
			doc_reader = lambda b: pd.read_csv(b, sep='\t')
		else:
			raise ValueError('File types in directory {} are inconsistent or invalid!'.format(dir_name))

		dfs = []
		for doc in dir_contents:
			df = doc_reader(os.path.join(dir_name + doc))
			if header:
				df = df[1:]
			dfs.append(df)

		self.df_docs = pd.concat(dfs)
		self.content_column_name = content_column_name

		self.phrases_to_join = phrases_to_join
		self.language = language
		self.phrases_to_remove = phrases_to_remove
		self.dont_stem = dont_stem

		self.min_word_length = min_word_length
		self.processing_to_do = processing_to_do

		punctuation_no_underscore = set(string.punctuation)
		punctuation_no_underscore.add('’')
		punctuation_no_underscore.add('”')
		punctuation_no_underscore.remove('_')
		self.punctuation = punctuation_no_underscore | extra_punctuation

		self.df_docs[self.content_column_name] = self.df_docs[self.content_column_name].apply(lambda b: b.lower())

		try:
			self.stopwords = stopwords.words(self.language)
		except OSError: 
			raise ValueError('No stopwords exist for language {}!'.format(self.language))

		self.stopwords += extra_stopwords

		preprocessing_completed = []
		if self.processing_to_do.get('remove_phrases', True):
			self.remove_phrases()
			preprocessing_completed.append('Removed phrases')

		if self.processing_to_do.get('join_phrases', True):
			self.join_phrases()
			preprocessing_completed.append('Joined phrases')

		if self.processing_to_do.get('remove_punctuation_and_digits', True):
			self.remove_punctuation_and_digits_and_tokenize()
			preprocessing_completed.append('Removed punctuation and digits')
		else:
			self.tokenize_content()
			preprocessing_completed.append('Tokenized content')

		if self.processing_to_do.get('remove_stopwords', True):
			self.remove_stopwords()
			preprocessing_completed.append('Removed stopwords')

		if self.processing_to_do.get('lemmatize_content', True):
			lemmatizer = WordNetLemmatizer()
			self.lemmatize_content(lemmatizer)
			preprocessing_completed.append('Lemmatized content')

		if self.processing_to_do.get('remove_short_words', True):
			self.remove_short_words(self.min_word_length)
			preprocessing_completed.append('Removed short words')

		self.preprocessing_completed = preprocessing_completed


	def what_preprocessing_was_completed(self):
		return sekf.preprocessing_completed

	def remove_phrases(self):
		if len(self.phrases_to_remove) == 0:
			raise ValueError('No phrases given for removal!')

		def remove_phrases_from_content(content:str) -> str:
			for w in self.phrases_to_remove:
				if w in content:
					content = re.sub(w, '', content)
	
			return content

		self.df_docs[self.content_column_name] = self.df_docs[self.content_column_name].apply(remove_phrases_from_content)

		return True

	def join_phrases(self):
		def join_phrases_in_content(content: str) -> str:
			for w in self.phrases_to_join:
				if w in content:
					content = re.sub(w,  '_'.join(w.split()), content)
			return content

		self.df_docs[self.content_column_name] = self.df_docs[self.content_column_name].apply(join_phrases_in_content)

		return True
		
	def remove_punctuation_and_digits_and_tokenize(self):
		def remove_punctuation_and_digits_from_content_and_tokenize(content: str) -> str:
			content = content.translate(str.maketrans('', '', ''.join(self.punctuation) + string.digits))
			content_ls = [word for word in content.split() if re.match('[a-zA-Z0-9]+', word)]
			return content_ls

		self.df_docs[self.content_column_name] = self.df_docs[self.content_column_name].apply(
			remove_punctuation_and_digits_from_content_and_tokenize
		)

		return True

	def tokenize_content(self):
		self.df_docs[self.content_column_name] = self.df_docs[self.content_column_name].apply(lambda b: [w for w in b.split()])

		return True

	def remove_stopwords(self):
		self.df_docs[self.content_column_name] = self.df_docs[self.content_column_name].apply(
			lambda content: [c for c in content if c not in self.stopwords]
		)

		return True

	def lemmatize_content(self, lemmatizer: WordNetLemmatizer):
		self.df_docs[self.content_column_name] = self.df_docs[self.content_column_name].apply(
			lambda content: [lemmatizer.lemmatize(c) for c in content if c not in self.dont_stem]
		)

		return True

	def remove_short_words(self, min_length: int):
		self.df_docs[self.content_column_name] = self.df_docs[self.content_column_name].apply(
			lambda content: [c for c in content if len(c) > 2]
		)

		return True

class LDAModeler(object):
	"""docstring for ClassName"""
	def __init__(
		self, content: Corpus, 
		low_bound_dict=0.02, 
		high_bound_dict=0.5
	):
		self.content = content
		self.my_corpus = list(self.content.df_docs[content.content_column_name])

		self.dictionary = corpora.Dictionary(self.my_corpus)
		self.dictionary.filter_extremes(no_below=low_bound_dict, no_above=high_bound_dict)
		self.corpus_bow = [self.dictionary.doc2bow(text) for text in self.my_corpus]

		self.lda_model = None
		self.num_topics = 0

	def model_topics(self, num_topics=10, num_keywords=20) -> tuple:
		self.num_topics = num_topics
		self.lda_model = models.wrappers.LdaMallet(
    		MALLET_PATH, 
    		corpus=self.corpus_bow, 
    		num_topics=num_topics, 
    		optimize_interval=10, 
    		id2word=self.dictionary, 
    		random_seed=1
		)
		topic_keywords = []
		for idx, topic in self.lda_model.show_topics(num_topics=num_topics, num_words=num_keywords, formatted=False):
			topic_keywords.append([w[0] for w in topic])

		topic_proportions = self.get_topic_proportions()
		topics_by_doc = self.lda_model.load_document_topics()

		return topic_keywords, topic_proportions, topics_by_doc

	def model_topics_to_spreadsheet(
		self, 
		num_topics=10, 
		num_keywords=20, 
		fname_keywords='topic_keywords_and_proportions.xlsx', 
		fname_topics_by_doc='topic_probabilities_by_document.xlsx',
		extra_df_columns_wanted=[]
	):
		topic_keyword_writer = pd.ExcelWriter(fname_keywords)
		doc_topic_writer = pd.ExcelWriter(fname_topics_by_doc)
		self.num_topics = num_topics
		topic_keywords, topic_proportions, topics_by_doc = self.model_topics(self.num_topics, num_keywords)

		topic_keywords_df = pd.DataFrame()
		for w_idx in range(num_keywords):
			topic_keywords_df['word_{}'.format(str(w_idx))] = [topic_keywords[i][w_idx] for i in range(len(topic_keywords))]
		topic_keywords_df['proportions'] = topic_proportions

		topic_dfs = []
		n_articles = ["n = " + str(len(self.corpus_bow))]
		topic_dfs.append(pd.Series(n_articles)) # number of articles
		topic_dfs.append(pd.Series(["\n"]))
		topic_dfs.append(topic_keywords_df.T)

		full_topic_df = pd.concat(topic_dfs)
		full_topic_df.to_excel(topic_keyword_writer)

		doc_topics = np.matrix([[m[1] for m in mat] for mat in topics_by_doc]) # create document --> topic proba matrix
		doc_max = [np.argmax(r) for r in doc_topics]
		doc_topic_df = self.content.df_docs[extra_df_columns_wanted + [self.content.content_column_name]]
		for c in range(self.num_topics):
			doc_topic_df['proba_topic_{}'.format(str(c))] = doc_topics[:, c]
		doc_topic_df['most_likely_topic'] = doc_max
		doc_topic_df.to_excel(doc_topic_writer)

		topic_keyword_writer.save()
		doc_topic_writer.save()

		return True

	def get_topic_proportions(self):
	    """
	    Given a corpus and a model and a number of topics, 
	    get the topic probability distribution for each document in the corpus 
	    and use it to get the average topic proportions in that corpus for the model
	    """
	    if self.num_topics == 0 or self.lda_model is None:
	    	raise ValueError('Number of topics not assigned and/or LDA model not trained!')

	    group_topic_proba = np.zeros(self.num_topics)
	    topics = self.lda_model[self.corpus_bow]
	    for td in topics:
	        group_topic_proba = group_topic_proba + np.array([t[1] for t in td])
	    z = group_topic_proba / sum(group_topic_proba)

	    return z


phrases_to_remove = [
    "subscribe",
    "utc",
    "chat with us in facebook messenger",
    "find out what's happening in the world as it unfolds",
    "get all the latest news on coronavirus and more delivered daily to your inbox.",
    "sign up here",
    "continue reading the main story",
    "get breaking news alerts and special reports",
    "the news and stories that matter, delivered weekday mornings",
    "a link has been posted to your facebook feed",
    "a link has been sent to your friend's email address",
]

phrases_to_join = [
'donald trump',
'white house',
'social distancing',
'stock market',
'new york',
'joe biden',
'bernie sanders',
'world health organization',
'face mask',
'mike pence',
'vice president',
'hong kong',
'united states',
'diamond princess',
'new hampshire',
'whats happening',
'lunar year',
'los angeles',
'san francisco',
'elissa slotkin']

dir_docs = '../../../Downloads/test_docs/'
my_corpus = Corpus(dir_docs, 'Unnamed: 3', phrases_to_remove=phrases_to_remove, phrases_to_join=phrases_to_join, header=True)
modeler = LDAModeler(my_corpus)
modeler.model_topics_to_spreadsheet()
