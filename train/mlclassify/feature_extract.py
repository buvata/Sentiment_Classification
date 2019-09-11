import re
import string
import emoji
import numpy as np
import scipy.sparse as sp
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion


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
            ('custom_features_pipeline', Pipeline([
                ('custom_features', FeatureUnion([
                    ('f01', NumWordsCharsFeature()),
                    ('f02', NumUpperLetterFeature()),
                    ('f03', NumLowerLetterFeature()),
                    ('f04', NumEmojiFeature())
                ], n_jobs=-1)),
                ('scaler', StandardScaler(with_mean=False))
            ])),
            ('word_features_pipeline', Pipeline([
                ('lowercase', Lowercase()),
                ('word_features', FeatureUnion([
                    ('with_tone', Pipeline([
                        ('remove_punct', RemovePunct()),
                        ('tf_idf_word', TfidfVectorizer(ngram_range=(1, 4), norm='l2', min_df=2))
                    ]))
                ], n_job=-1)),
            ]))
        ], n_job=-1)),
        ('alg', SVC(kernel='linear', C=0.2175, class_weight=None, verbose=True))
        ])


clf_char = Pipeline([
        ('remove_spaces', RemoveMutilSpaces()),
        ('features', FeatureUnion([
            ('custom_features_pipeline', Pipeline([
                ('custom_features', FeatureUnion([
                    ('f01', NumWordsCharsFeature()),
                    ('f02', NumUpperLetterFeature()),
                    ('f03', NumLowerLetterFeature()),
                    ('f04', NumEmojiFeature())
                ], n_jobs=-1)),
                ('scaler', StandardScaler(with_mean=False))
            ])),
            ('char_features_pipeline', Pipeline([
                ('lowercase', Lowercase()),
                ('char_features', FeatureUnion([
                    ('with_tone', Pipeline([
                        ('remove_punct', RemovePunct()),
                        ('tf_idf_char', TfidfVectorizer(ngram_range=(1, 6), norm='l2', min_df=2, analyzer='char')),
                        ('tf_idf_char_wb', TfidfVectorizer(ngram_range=(1,6), norm='l2', min_df=2,analyzer='char_wb'))
                    ]))
                ], n_job=-1)),
            ]))
        ], n_job=-1)),
        ('alg', SVC(kernel='linear', C=1, class_weight=None, verbose=True))
        ])


def smv_classify(clf_word, clf_char, train_data, mode):
    if mode == 'word':
        clf_word.fit(train_data.data, train_data.target)
    if mode == "char":
        clf_char.fit(train_data.data, train_data.target)
