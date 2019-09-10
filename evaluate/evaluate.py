from __future__ import absolute_import

import matplotlib.pyplot as plt 
import scikitplot as skplt
import pandas as pd 
import numpy as np 
import train.config as cf 
from train.utils import *
from train.model.lstm_char_word import LSTMWordChar
import torch
from torchtext import vocab
from torchtext import data
from sklearn.metrics import confusion_matrix, classification_report, f1_score


def get_input_processor_words(inputs, vocab_word, vocab_char=None):
   
    inputs_word = data.Field(init_token="<SOS>", eos_token="<EOS>", batch_first=True, lower=True)
   
    inputs_word.vocab = vocab_word
   
    if vocab_char is not None:
        inputs_char_nesting = data.Field(tokenize=list, init_token="<SOS>", eos_token="<EOS>", 
                                        batch_first=True, fix_length=cf.lstm_char_word['fix_length_char'])

        inputs_char = data.NestedField(inputs_char_nesting, 
                                        init_token="<SOS>", eos_token="<EOS>")
 
        inputs_char.vocab = inputs_char_nesting.vocab = vocab_char
        
        fields = [(('word', 'char'), (inputs_word, inputs_char))]
    else:
        fields = [('word', inputs_word)]

    if not isinstance(inputs, list):
        inputs = [inputs]

    examples = []
   
    for line in inputs:
        examples.append(data.Example.fromlist([line], fields))
 
    dataset = data.Dataset(examples, fields)
    batchs = data.Batch(data=dataset, 
                            dataset=dataset,
                            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) 
   
    # Entire input in one batch
    return data.Batch(data=dataset, 
                      dataset=dataset,
                      device=None)


def predict_sentiment(model, cf, test_iters):
  
    test_preds = []
    test_weight_preds = []
    for batch in test_iters:
        preds = model(batch)
        preds = torch.sigmoid(preds)

        w_pred = preds.data.numpy()      # weight_predict 
        y_pred = torch.max(preds, 1)[1].numpy()  # return label
       
        test_preds.append(y_pred)
        
        test_weight_preds.append(w_pred)

    test_preds = [item for sublist in test_preds for item in sublist]
    test_weight_preds = [item for sublist in test_weight_preds for item in sublist]

    return test_preds, test_weight_preds


def error_predict(model, cf, test_iters):
    data_test = pd.read_csv(cf.data['test'])
   
    test_preds, test_weight_preds = predict_sentiment(model, cf, test_iters)
   
    indices = [i for i in range(len(test_preds)) if test_preds[i] != data_test.label[i]]
    error_predictions = data_test.iloc[indices,:]
    error_predictions.insert(3, "weight", [test_weight_preds[i] for i in indices], True)
    error_predictions.to_csv("../model_results/error_predict.csv")
    return error_predictions


def plot_confusion_matrix(model, cf):
    vocab_word, vocab_char = load_vocab(cf)
   
    print("len_vocab_word", len(vocab_word))

    data_test = pd.read_csv(cf.data['test'])
    input_texts = data_test.combine_comment.tolist()
    
    test_iters = []
    for i in range(len(input_texts)//32+1):
        input_text = input_texts[i*32:(i+1)*32]
        test_iter = get_input_processor_words(input_text, vocab_word, vocab_char)
        test_iters.append(test_iter)
        
    y_test = data_test.label.tolist()
    y_pred, _ = predict_sentiment(model, cf, test_iters)

    error_predict(model, cf, test_iters)

    labels = ['pos', 'neg']
   
    print(classification_report(y_test, y_pred, target_names=labels))
   
    skplt.metrics.plot_confusion_matrix(
            y_test, 
            y_pred,
            figsize=(6, 6))
    plt.show()