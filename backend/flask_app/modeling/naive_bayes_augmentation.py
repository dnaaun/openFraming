import collections
import itertools
import typing as T

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import sklearn
from sklearn.naive_bayes import MultinomialNB


class NaiveBayesDatasetCreator(object):
    def __init__(
        self, keywords_fname, labeled_dset_fname, unlabeled_dset_fname, label_list
    ):
        self.keywords_df = pd.read_excel(keywords_fname, header=[0, 1, 2])
        self.keywords_df.drop(0)
        self.keywords = list(
            set(
                itertools.chain.from_iterable(
                    [[v for v in vv[1:]] for vv in self.keywords_df.values[:-2]]
                )
            )
        )
        print(self.keywords)
        self.lemmatizer = WordNetLemmatizer()

        def lower_split_lemmatize(txt, lemmatizer, keywords):
            txt = txt.lower()
            txt = txt.split()
            txt = collections.Counter(lemmatizer.lemmatize(t.strip()) for t in txt)
            return np.array([txt.get(k, 0) for k in keywords])

        self.labeled_dset = pd.read_excel(labeled_dset_fname)
        self.labeled_dset_mtx = [
            lower_split_lemmatize(b, self.lemmatizer, self.keywords)
            for b in self.labeled_dset["example"].tolist()
        ]

        self.unlabeled_dset = pd.read_excel(unlabeled_dset_fname)
        self.unlabeled_dset_mtx = [
            lower_split_lemmatize(b, self.lemmatizer, self.keywords)
            for b in self.unlabeled_dset["example"].tolist()
        ]

        self.labeled_dset_cats = np.array(self.labeled_dset["category"])
        self.label_list = label_list
        self.priors = collections.Counter(self.labeled_dset_cats)
        tot_entries = len(self.labeled_dset_cats)
        self.priors = {L: self.priors[L] / tot_entries for L in self.label_list}

    def make_fake_dataset(self, throttling_param, alpha, result_fname):
        clf = MultinomialNB(alpha=alpha)
        # print(self.labeled_dset_mtx)
        print(np.mean([len(a) for a in self.labeled_dset_mtx]))
        print(self.labeled_dset_cats)
        clf.fit(self.labeled_dset_mtx, self.labeled_dset_cats)

        to_take = self.labeled_dset

        cats_take = {k: [] for k in self.label_list}
        cats_pred = clf.predict_proba(self.unlabeled_dset_mtx)
        best_cat = [np.argmax(c) for c in cats_pred]
        best_proba = [max(c) for c in cats_pred]

        for i, tup in enumerate(zip(best_cat, best_proba)):
            cat = tup[0]
            proba = tup[1]
            cats_take[self.label_list[cat]] = cats_take[self.label_list[cat]] + [
                (proba, i)
            ]

        unlabeled_take_ix = []
        best_global = []
        for cat, probas in cats_take.items():
            top_k = int(throttling_param - self.priors[cat] * len(self.priors))
            best = sorted(probas, key=lambda b: b[0], reverse=True)[:top_k]
            unlabeled_take_ix = unlabeled_take_ix + [b[1] for b in best]
            best_global = best_global + [(cat, b[1]) for b in best]
        unlabeled_take_df = self.unlabeled_dset.iloc[unlabeled_take_ix]
        unlabeled_take_df["category"] = [
            b[0] for b in sorted(best_global, key=lambda b: b[1])
        ]
        new_df = pd.concat(
            [
                to_take[["example", "category"]],
                unlabeled_take_df[["example", "category"]],
            ]
        )
        new_df.to_csv(result_fname)
        print(len(new_df))
        print(len(to_take))
        return True
