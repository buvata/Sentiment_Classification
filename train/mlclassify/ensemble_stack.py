import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC 
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from vecstack import stacking
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lbg 
import unidecode 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
import random
from sklearn import preprocessing
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras import optimizers
from sklearn.model_selection import train_test_split  
from keras.utils import to_categorical

tfidf = TfidfVectorizer(
    ngram_range=(1, 5),
    min_df=5,
    analyzer='char',
    max_df=0.8,
    sublinear_tf=True
)


def load_data(filename_train, filename_test):
    train = pd.read_csv(filename_train)
    test = pd.read_csv(filename_test)
  
    train_comments = train['combine_comment'].fillna("none").values
    test_comments = test['combine_comment'].fillna("none").values

    y_train = train['label'].values
    y_test = test['label'].values
    return train_comments, test_comments, y_train, y_test


def train(train_comments, y_train):

    X_train = tfidf.fit_transform(train_comments)

    models = [
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        xgb.XGBClassifier(),
        lbg.LGBMClassifier()
    ]
    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=CV)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

    return cv_df


def stacking_model(y_train):
    train = pd.read_csv(filename_train)
    train_comments = train['combine_comment'].fillna("none").values
    X1_train, X1_test, y1_train, y1_test = train_test_split(train_comments, y_train, test_size=0.2, random_state=42)   
    
    # first 
    pipe_RF = Pipeline([
                     ('tfidf_vectorizer', tfidf),
                     ('clf', RandomForestClassifier())
                    ])

    pipe_GB = Pipeline([
                        ('tfidf_vectorizer', tfidf),
                        ('clf', GradientBoostingClassifier(random_state=0, learning_rate=0.3, 
                                n_estimators=100, max_depth=5))
                        ])

    pipe_XGB = Pipeline([ 
                        ('tfidf_vectorizer', tfidf),
                        ('clf', xgb.XGBClassifier(random_state=0, learning_rate=0.2, 
                                n_estimators=100, max_depth=5))
                        ])

    # List of pipelines
    pipelines = [pipe_RF, pipe_GB, pipe_XGB]
    pipeline_names = ['Random Forest', 'GradientBoost', "XGBoost"]

    # Loop to fit each of the three pipelines
    for pipe in pipelines:
        print(pipe)
        pipe.fit(X1_train, y1_train)

    # Compare accuracies
    X1_scores = []
    for index, val in enumerate(pipelines):
        tup = (pipeline_names[index], val.score(X1_test, y1_test), val.predict_proba(X1_train), val.predict(X1_train))
        X1_scores.append(tup)
        print('%s pipeline test accuracy: %.3f' % (pipeline_names[index], val.score(X1_test, y1_test)))

    classes = ['pos', 'neg']
    R1_AVG_Scores = (X1_scores[0][2] + X1_scores[1][2] + X1_scores[2][2])/3
    R1_df = pd.DataFrame(R1_AVG_Scores, columns=[(item + "_AVG") for item in classes])
    
    # second
    X2_train, X2_test, y2_train, y2_test = train_test_split(R1_df, y1_train, test_size=0.2, random_state=123)  
    y2_test = to_categorical(y2_test)
    y2_train=to_categorical(y2_train)

    # Neural Networks Model
    random.seed(123)
    model = models.Sequential()
    model.add(layers.Dense(8, input_dim=2, kernel_initializer='normal', activation='relu')) #2 hidden layers
    model.add(layers.Dense(5, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(optimizer='SGD',
                loss='binary_crossentropy',
                metrics=['accuracy'])
                
    model_val = model.fit(X2_train, y2_train,
                epochs=20,
                batch_size=32,
                validation_data=(X2_test, y2_test))
    return model_val


def plot_loss(model_val):
    plt.plot(model_val.history['acc'])
    plt.plot(model_val.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(model_val.history['loss'])
    plt.plot(model_val.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == "__main__":
   
    filename_train = "../data/data_train.csv"
    filename_test = "../data/data_test.csv"
    train_comments, test_comments, y_train, y_test = load_data(filename_train, filename_test)

    cv_df = train(train_comments, y_train)

    print(cv_df.head())

    model = stacking_model(y_train)
    plot_loss(model)