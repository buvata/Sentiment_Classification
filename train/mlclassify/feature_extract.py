import re
import emoji
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


class RemovePunct(BaseEstimator, TransformerMixin):
    non_special_chars = re.compile('[^A-Za-z0-9 ]+')

    def remove_punct(self, s):
        return re.sub(self.non_special_chars, '', s)

    def transform(self, x):
        return [self.remove_punct(s) for s in x]

    def fit(self, x, y=None):
        return self


class RemoveMutilSpaces(BaseEstimator, TransformerMixin):
    def remove_consecutive_spaces(self, s):
        return ' '.join(s.split())

    def transform(self, x):
        return [self.remove_consecutive_spaces(s) for s in x]

    def fit(self, x, y=None):
        return self


class Lowercase(BaseEstimator, TransformerMixin):
    def transform(self, x):
        return [s.lower() for s in x]

    def fit(self, x, y=None):
        return self


class NumUpperLetterFeature(BaseEstimator, TransformerMixin):
    def count_upper(self, s):
        n_uppers = sum(1 for c in s if c.isupper())
        return n_uppers / (1 + len(s))

    def transform(self, x):
        counts = [self.count_upper(s) for s in x]
        return sp.csr_matrix(counts, dtype=np.float64).transpose()

    def fit(self, x, y=None):
        return self


class NumLowerLetterFeature(BaseEstimator, TransformerMixin):
    def count_upper(self, s):
        n_uppers = sum(1 for c in s if c.islower())
        return n_uppers / (1 + len(s))

    def transform(self, x):
        counts = [self.count_upper(s) for s in x]
        return sp.csr_matrix(counts, dtype=np.float64).transpose()

    def fit(self, x, y=None):
        return self


class NumWordsCharsFeature(BaseEstimator, TransformerMixin):
    def count_char(self, s):
        return len(s)

    def count_word(self, s):
        return len(s.split())

    def transform(self, x):
        count_chars = sp.csr_matrix([self.count_char(s) for s in x], dtype=np.float64).transpose()
        count_words = sp.csr_matrix([self.count_word(s) for s in x], dtype=np.float64).transpose()

        return sp.hstack([count_chars, count_words])

    def fit(self, x, y=None):
        return self


class NumEmojiFeature(BaseEstimator, TransformerMixin):
    def count_emoji(self, s):
        emoji_list = []
        for c in s:
            if c in emoji.UNICODE_EMOJI:
                emoji_list.append(c)
        return len(emoji_list) / (1 + len(s.split()))

    def transform(self, x):
        counts = [self.count_emoji(s) for s in x]
        return sp.csr_matrix(counts, dtype=np.float64).transpose()

    def fit(self, x, y=None):
        return self


clf_word = Pipeline([
        ('remove_spaces', RemoveMutilSpaces()),
        ('features', FeatureUnion([
            ('word_features_pipeline', Pipeline([
                ('lowercase', Lowercase()),
                ('word_features', FeatureUnion([
                    ('with_tone', Pipeline([
                        ('remove_punct', RemovePunct()),
                        ('tf_idf_word', TfidfVectorizer(ngram_range=(1, 4), norm='l2', min_df=2))
                    ]))
                ], n_jobs=-1)),
            ]))
        ], n_jobs=-1)),
        ('alg', SVC(kernel='linear', C=0.2175, class_weight=None, verbose=True))
        ])


clf_char = Pipeline([
        ('remove_spaces', RemoveMutilSpaces()),
        ('features', FeatureUnion([
            ('char_features_pipeline', Pipeline([
                ('lowercase', Lowercase()),
                ('char_features', FeatureUnion([
                    ('with_tone', Pipeline([
                        ('remove_punct', RemovePunct()),
                        ('tf_idf_char', TfidfVectorizer(ngram_range=(1, 6), norm='l2', min_df=2, analyzer='char')),
                        ('tf_idf_char_wb', TfidfVectorizer(ngram_range=(1, 6), norm='l2', min_df=2, analyzer='char_wb'))
                    ]))
                ], n_jobs=-1)),
            ]))
        ], n_jobs=-1)),
        ('alg', SVC(kernel='linear', C=1, class_weight=None, verbose=True))
        ])


pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    # ('clf', SVC(kernel='linear')),
    ('clf', RandomForestClassifier())
])


def feature_importance(train_data, target):
    cv = CountVectorizer()
    cv.fit(train_data)
    vocab = cv.get_feature_names()
    params = {'clf__n_estimators': (10, 100)}
    optimized_svm = GridSearchCV(pipeline,
                                 param_grid=params,
                                 cv=3,
                                 n_jobs=-1)

    optimized_svm.fit(train_data, target)
    feature_importance = optimized_svm.best_estimator_.named_steps['clf'].feature_importances_
    return feature_importance, vocab


def plot_feature_importance(feature_importance, list_vocabulary, n_feature_show=50):
    # list vocabulary: ['an_ninh', 'an_toàn', 'apec', 'ban'...]
    # list_feature_importance: [0, 0.124, 0, 0]
    index_sorted_important_ft = sorted(range(len(feature_importance)), key=feature_importance.__getitem__, reverse=True)
    list_best_important_ft = index_sorted_important_ft[0: n_feature_show]

    # get list probability of best ft:
    # now get list mapping vocabulary with number count
    list_best_ft_vocabulary = []
    list_best_ft_probability = []
    for e_index in list_best_important_ft:
        list_best_ft_vocabulary.append(list_vocabulary.__getitem__(e_index))
        list_best_ft_probability.append(feature_importance.__getitem__(e_index))

    name_proba_ft_important = list(zip(list_best_ft_vocabulary, list_best_ft_probability))
    df_name_proba = pd.DataFrame(name_proba_ft_important, columns=['name_ft', 'probability'])
    df_name_proba.to_csv("feature_importance_show.csv")

    seaborn.scatterplot(x="name_ft", y="probability", data=df_name_proba)
    plt.show()


def smv_classify(clf_word, clf_char, train_data, mode):
    if mode == 'word':
        clf_word.fit(train_data.comment, train_data.label)

    if mode == "char":
        clf_char.fit(train_data.comment, train_data.label)


if __name__ == '__main__':
    data_train = pd.read_csv("../../data/data_model/data_train.csv")
    train_data = data_train.comment
    target = data_train.label

    feature_importances, vocab = feature_importance(train_data, target)
    plot_feature_importance(feature_importances, vocab, n_feature_show=20)