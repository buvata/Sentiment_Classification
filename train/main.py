from __future__ import absolute_import
import os 
import sys
import pandas as pd
import config as cf
import torch
import torch.nn as nn 
sys.path.insert(0,"../")
from evaluate.evaluate import *
from utils import * 
from data.dataloader import *
from model.lstm_char_word import LSTMWordChar
from model.cnn_sent import CNN1dWord
from model.lstm_cnn_char_word import LSTMCNNWordChar


def train_network(cf, model, train, valid): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train.init_epoch()
    valid.init_epoch()

    if cf.model_train['save_device'] == 'cpu':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)

    for param in model.parameters():
        param.requires_grad = True 
    optimizer = torch.optim.Adam(model.parameters(), lr = cf.model_train['learning_rate'])
    losses, train_acc, valid_acc, loss = [], [], [], 0
    epochs = []
    best_valid_acc = 0.
    best_epoch = 0 
    for epoch in range(cf.model_train['num_epochs']):
        print(epoch)
        for i, batch in enumerate(train): 
            l = batch.label   
            optimizer.zero_grad()
            pred = model.compute_forward(batch)
            loss = model.loss(batch)  
            loss.backward()  
            optimizer.step()
        
        losses.append(float(loss))

        epochs.append(epoch)
        train_acc.append(model.get_accuracy(train))
        valid_acc.append(model.get_accuracy(valid))

        if valid_acc[-1] > best_valid_acc:
            best_valid_acc = valid_acc[-1]
            best_epoch = epoch 

            if cf.model_train == 'lstm_word_char':
                if cf.model_train['save_model']:
                    cf.train_mode['modelfname'] = "lstm_word_char" + \
                    "_".join(["_use_cnn", str(cf.lstm_char_word['use_char_cnn'])]) + \
                    "_".join(["_w_emb", str(cf.lstm_char_word['word_embedding_dim'])]) + \
                    "_".join(["_lr", str(cf.model_train['learning_rate'])]) + \
                    "_".join(["_bs", str(cf.model_train['batch_size'])]) + \
                    "_".join(["_hidden_size_w", str(cf.lstm_char_word['hidden_size_word'])]) + \
                    ".pt"
                    model.save(cf.model_train['modelfnamepath'] + cf.model_train['modelfname'])

            if cf.model_train == 'lstm_cnn_word_char':
                if cf.model_train['save_model']:
                    cf.train_mode['modelfname'] = "lstm_cnn_word_char" + \
                    "_".join(["_use_cnn", str(cf.lstm_cnn_char_word['use_char_cnn'])]) + \
                    "_".join(["_w_emb", str(cf.lstm_cnn_char_word['word_embedding_dim'])]) + \
                    "_".join(["_hidden_size_w", str(cf.lstm_cnn_char_word['hidden_size_word'])]) + \
                    "_".join(["_c_emb", str(cf.lstm_cnn_char_word['char_embedding_dim'])]) + \
                    "_".join(["_n_filter", str(cf.lstm_cnn_char_word['char_cnn_number_filter'])]) + \
                    "_".join(["window_size", str(cf.lstm_cnn_char_word['char_window_size'])]) + \
                    "_".join(["_lr", str(cf.model_train['learning_rate'])]) + \
                    "_".join(["_bs", str(cf.model_train['batch_size'])]) + \
                    ".pt"

                    model.save(cf.model_train['modelfnamepath'] + cf.model_train['modelfname'])

            if cf.model_train == 'cnn_word':
                if cf.model_train['save_model']:
                    cf.train_mode['modelfname'] = "cnn_word" + \
                    "_".join(["_filter_num", str(cf.cnn_word['cnn_filter_num'])]) + \
                    "_".join(["_kernel_size", str(cf.cnn_word['window_size'])]) + \
                    "_".join(["_w_emb", str(cf.cnn_word['embedding_dim'])]) + \
                    "_".join(["_lr", str(cf.model_train['learning_rate'])]) + \
                    "_".join(["_bs", str(cf.model_train['batch_size'])]) + \
                    ".pt"
                    model.save(cf.model_train['modelfnamepath'] + cf.model_train['modelfname']) 

        print("Epoch %d; Loss %f; Train Acc %f ; Val Acc %f " % (
            epoch+1, loss, train_acc[-1], valid_acc[-1]))


    if cf.model_train['save_model']:
        if not os.path.isfile(cf.model_train['resultsdfpath']):
            results_df = pd.DataFrame(columns = ["modelname", "best_acc", "best_epoch"])
            experiment_df = pd.DataFrame([[cf.model_train['modelfname'], best_valid_acc, best_epoch]],
                columns = ["modelname", "best_acc", "best_epoch"])
            results_df = results_df.append(experiment_df, ignore_index=True)
            results_df.to_pickle(cf.model_train['resultsdfpath'])
        else:
            results_df = pd.read_pickle(cf.model_train['resultsdfpath'])
            experiment_df = pd.DataFrame([[cf.model_train['modelfname'], best_valid_acc, best_epoch]],
                columns = ["modelname", "best_acc", "best_epoch"])
            results_df = results_df.append(experiment_df, ignore_index=True)
            results_df.to_pickle(cf.model_train['resultsdfpath'])


if __name__ == "__main__":
   
    train_iter, vald_iter, trainds, valds, txt_field, char_field = get_dataloader_word_char(cf)
    vocab_word, vocab_char = load_vocab(cf)
    build_vocab(txt_field, trainds)
    build_vocab(char_field, trainds)

    vocab_word = txt_field.vocab
    vocab_char = char_field.vocab    

    vocab_size = len(vocab_word)
    pad_idx = vocab_word.stoi[txt_field.pad_token]

    if cf.model_train['mode'] == 'lstm_word_char': 
        model = LSTMWordChar(vocab_word, vocab_char, cf )

    if cf.model_train['mode'] == 'cnn_word_char':
        model = CNN1dWord(cf, vocab_size, pad_idx)

    if cf.model_train['mode'] == 'lstm_cnn_word_char' :
        train_cnn , vald_cnn , trainds, valds, txt_field, char_field = get_dataloader_cnn_word_char(cf)
        build_vocab(txt_field, trainds)
        build_vocab(char_field, trainds)
        vocab_word = txt_field.vocab
        vocab_char = char_field.vocab   

        model = LSTMCNNWordChar(vocab_word, vocab_char, cf)

  
    train_mode = True
    if train_mode is True :
        train_network(cf, model, train_cnn, vald_cnn ) 
    else:
        model_test = load_model(cf, model, '../model_results/model_cf_use_cnn_True_w_emb_100_lr_0.0001_bs_32_hidden_size_w_16.pt')
        plot_confusion_matrix(model_test, cf) 

    

    








