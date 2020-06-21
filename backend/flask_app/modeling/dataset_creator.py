import typing as T

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from lda import EXPERT_LABEL_COLUMN_NAME, ID_COLUMN_NAME, TOPIC_PROBA_PREFIX


class SemiSupervisedDatasetCreator(object):
    def __init__(
        self,
        doc_topic_proportions_fname: str,
        labeled_dset_fname: str,
        label_list: T.List[str],
        similarity_threshold: float = 0.0001,
    ):
        """
        note that we're assuming both files are Excel dataframes, 
        as we're giving them to the user in that format.
        """
        self.df_topic_proportions = pd.read_excel(doc_topic_proportions_fname)
        self.df_labeled = pd.read_excel(labeled_dset_fname)
        print(self.df_topic_proportions["OBJECT_ID"])
        self.df_labels_minimal = self.df_labeled[
            [EXPERT_LABEL_COLUMN_NAME, ID_COLUMN_NAME]
        ]
        self.df_topic_joint = self.df_topic_proportions.merge(
            self.df_labels_minimal,
            left_on=ID_COLUMN_NAME,
            right_on=ID_COLUMN_NAME,
            how="outer",
        )
        print(len(self.df_topic_joint))
        print(len(self.df_labels_minimal), len(self.df_topic_proportions))
        self.labels = label_list

        self.topic_proba_cols = sorted(
            [c for c in self.df_topic_proportions.columns if TOPIC_PROBA_PREFIX in c]
        )
        self.throttling_parameter = 60
        self.similarity_threshold = similarity_threshold

    def get_cosine_dist_reliability(self, line, centroids):
        vec = line[self.topic_proba_cols].values
        dists = [
            (label, cosine_similarity([centroids[label]], [vec])[0])
            for label in self.labels
        ]
        sorted_dists = sorted(dists, key=lambda b: b[1])
        reliability = sorted_dists[1][1] - sorted_dists[0][1]

        return (reliability, sorted_dists[0][0])

    def do_reliability_iteration(self, df_topic_joint):
        grs_by_label = df_topic_joint.groupby(EXPERT_LABEL_COLUMN_NAME)

        centroids = {}
        rs = {}
        for label, gr in grs_by_label:
            centroids[label] = gr[self.topic_proba_cols].mean().values
            rs[label] = len(gr)

        print(centroids)
        unlabeled = df_topic_joint[df_topic_joint[EXPERT_LABEL_COLUMN_NAME].isna()][
            [ID_COLUMN_NAME] + self.topic_proba_cols
        ]

        min_r = min(rs.values())
        rs = {k: v / min_r for k, v in rs.items()}
        unlabeled["placeholder"] = unlabeled.apply(
            lambda b: self.get_cosine_dist_reliability(b, centroids), axis=1
        )
        unlabeled["reliability"] = unlabeled["placeholder"].apply(lambda b: b[0])
        unlabeled["best_label"] = unlabeled["placeholder"].apply(lambda b: b[1])
        print(np.max(unlabeled["reliability"]))
        if self.similarity_threshold > np.max(unlabeled["reliability"]):
            return df_topic_joint, False
        else:
            for label in self.labels:
                num_to_get = int(self.throttling_parameter - rs[label])
                best_in_label = unlabeled[unlabeled["best_label"] == label].sort_values(
                    by="reliability"
                )[ID_COLUMN_NAME]
                ids_to_label = set(best_in_label[:num_to_get])
                df_topic_joint[EXPERT_LABEL_COLUMN_NAME].loc[
                    df_topic_joint[ID_COLUMN_NAME].isin(ids_to_label)
                ] = label
            return df_topic_joint, True

    def get_labeled_df(self):
        keep_going = True
        while keep_going:
            # print(self.df_topic_joint)
            self.df_topic_joint, keep_going = self.do_reliability_iteration(
                self.df_topic_joint
            )
            print("sum", sum(np.isnan(self.df_topic_joint[EXPERT_LABEL_COLUMN_NAME])))

        return self.df_topic_joint


# from lda import Corpus, LDAModeler

# dir_docs = (
#     "../../../../Downloads/bu_com_climate_change_tweets_2063_deduplicated_splits/0/"
# )
# my_corpus = Corpus(
#     dir_docs,
#     "tweet",
#     "GUID",
#     header=True,
#     processing_to_do={"remove_phrases": False, "join_phrases": False},
# )
# modeler = LDAModeler(my_corpus)
# modeler.model_topics_to_spreadsheet(3, 20)
# my_corpus.random_sample(100, "./samples.xlsx", ["sentiment"])

# s = SemiSupervisedDatasetCreator(
#     "./topic_probabilities_by_document.xlsx",
#     "../../../../Downloads/samples.xlsx",
#     [1, 2, 99],
# )

# df = s.get_labeled_df()
# gold_tr = pd.read_csv(
#     "../../../../Downloads/bu_com_climate_change_tweets_2063_deduplicated_splits/0/train.csv"
# )
# gold_ts = pd.read_csv(
#     "../../../../Downloads/bu_com_climate_change_tweets_2063_deduplicated_splits/0/dev.csv"
# )
# gold = pd.concat([gold_tr, gold_ts])
# truth = {guid: sentiment for guid, sentiment in zip(gold["GUID"], gold["sentiment"])}
# guesses = {
#     guid: sentiment
#     for guid, sentiment in zip(df[ID_COLUMN_NAME], df[EXPERT_LABEL_COLUMN_NAME])
# }

# correct = 0.0
# wrong = 0.0
# print(len(truth), len(guesses))
# print(set(guesses.keys()).issubset(set(truth.keys())))
# for my_id in truth.keys():
#     if my_id not in guesses:
#         print(my_id)
#         continue
#     if truth[my_id] == guesses[my_id]:
#         correct += 1.0
#     else:
#         wrong += 1.0

# print(correct / (correct + wrong))
