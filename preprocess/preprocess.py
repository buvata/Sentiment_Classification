import re 
import string
import numpy as np 
import pandas as pd 
from sklearn import utils 
import emoji
import unidecode 
import unicodedata
from sklearn.model_selection import train_test_split
from text_handler import *

regex_train = "train_[0-9]*[\s\S]*?\"\n[0|1]"
regex_test = "test_[0-9]*[\s\S]*?\"\n"

# Load train data
def load_train_data(filename):
    with open(filename, "r") as f:
        lines = f.read()

    # Find trainning data 
    train = re.findall(regex_train, lines)

    # Split to ids, labels, comments
    train_ids = [t.split("\n")[0] for t in train]
    train_labels = [t.split("\n")[-1] for t in train]
    train_comments = ["\n".join(t.split("\n")[1:-1]) for t in train]
    train_comments = [t[1:-1] for t in train_comments]
    assert len(train_ids) == len(train_labels) == len(train_comments)

    # Create dataframe
    train_df = pd.DataFrame(
        {
            "id": train_ids,
            "comment": train_comments,
            "label": train_labels
        }
    )

    # Save
    train_df.to_csv("./train.csv", index=False) 
    return train_df

# Load test data
def load_test_data(filename):
    with open(filename, "r") as f:
        lines = f.read()

    # Find test data 
    test = re.findall(regex_test, lines)

    # Split to ids, labels, comments
    test_ids = [t.split("\n")[0] for t in test]
    test_comments = ["\n".join(t.split("\n")[1:]) for t in test]
    test_comments = [t[1:-2] for t in test_comments]
    assert len(test_ids) == len(test_comments)

    # Create dataframe
    test_df = pd.DataFrame(
        {
            "id": test_ids,
            "comment": test_comments
        }
    )
    # Save
    test_df.to_csv("./test.csv", index=False)
    return test_df


'''
def count_num_emoji(df):
    str(df.label)
    good_df = df[df['label'] == '0']
    good_comment = good_df['comment'].values
    good_emoji = []
    for c in good_comment:
        good_emoji += extract_emojis(c)

    good_emoji = np.unique(np.asarray(good_emoji))

    bad_df = df[df['label'] == '1']
    bad_comment = bad_df['comment'].values

    bad_emoji = []
    for c in bad_comment:
        bad_emoji += extract_emojis(c)

    bad_emoji = np.unique(np.asarray(bad_emoji))

    comment = df['comment']
    n_good_emoji = 0
    n_bad_emoji = 0
    for c in comment:
        if c in good_emoji:
            n_good_emoji += 1
        if c in bad_emoji:
            n_bad_emoji += 1
    
    df['n_good_emoji'] = n_good_emoji
    df['n_bad_emoji'] = n_bad_emoji

    return df
'''


def add_feature(df):
    df['comment'] = df['comment'].astype(str).fillna(' ')
    df['comment'] = df['comment'].str.lower()
    df['comment'] = df['comment'].apply(lambda s: remove_punctuation(s))
    df['comment'] = df['comment'].apply(lambda s: remove_multiple_space(s))
    df['comment_w_tone'] = df['comment'].apply(lambda s: remove_tone(s))
    df['comment_w_tone'] = df['comment_w_tone'].apply(lambda s: remove_null_element(s))
    df['combine_comment'] = df['comment'] + " " +df['comment_w_tone'].astype(str)
    return df 

if __name__ == "__main__":
    data = load_train_data("../data/train.crash")
    #data = count_num_emoji(data)
    data = add_feature(data)
    train, validate, test = np.split(data.sample(frac=1), [int(.6*len(data)), int(.8*len(data))])
    train.to_csv("../data/data_train.csv",index=False)
    validate.to_csv("../data/data_validate.csv",index=False)
    test.to_csv("../data/data_test.csv",index=False)
   


