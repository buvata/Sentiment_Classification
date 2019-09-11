import numpy as np
import os
import torchtext
import heapq
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import config as cf 
from utils import * 
from sub_layer.cnn_feature_extract import CNNFeatureExtract
from sub_layer.highway import Highway


class LSTMCNNWordChar(nn.Module):
    def __init__(self, vocab_word, vocab_char, cf):
        
        super(LSTMCNNWordChar, self).__init__()

        self.num_classes = cf.lstm_cnn_char_word['num_classes']
        self.word_embedding_dim = cf.lstm_cnn_char_word['word_embedding_dim']
        self.hidden_size_word = cf.lstm_cnn_char_word['hidden_size_word']

        self.char_embedding_dim = cf.lstm_cnn_char_word['char_embedding_dim'] 
        self.hidden_size_char = cf.lstm_cnn_char_word['hidden_size_char']   
        
        self.vocab_word = vocab_word
        self.vocab_char = vocab_char

        self.word_embedding_layer = nn.Embedding(len(vocab_word), self.word_embedding_dim)
        self.char_embedding_layer = None

        self.use_cnn_feature_char = cf.lstm_cnn_char_word['use_cnn_feature_char']
        if self.use_cnn_feature_char:
            self.char_cnn_filter_num = cf.lstm_cnn_char_word['char_cnn_filter_num']
            self.char_window_size = cf.lstm_cnn_char_word['char_window_size']
            self.dropout_cnn = cf.lstm_cnn_char_word['dropout_cnn']

        if vocab_char is not None and self.char_embedding_dim > 0:
            self.char_embedding_layer = nn.Embedding(len(vocab_char), self.char_embedding_dim)

            self.layer_char_cnn = CNNFeatureExtract(self.char_embedding_dim,
                                                    self.char_cnn_filter_num,
                                                    self.char_window_size,
                                                    self.dropout_cnn)
                
        each_number_output_filter = self.char_cnn_filter_num * len(self.char_window_size) 
        self.hidden_size_char = each_number_output_filter
      
        if vocab_word.vectors is not None:
            if self.word_embedding_dim != vocab_word.vectors.shape[1]:
                raise ValueError("expect embedding word: {} but got {}".format(self.word_embedding_dim,
                                                                               vocab_word.vectors.shape[1]))

            self.word_embedding_layer.weight.data.copy_(vocab_word.vectors)
            self.word_embedding_layer.requires_grad = False

        # lstm get input shape (seq_len, batch_size, input_dim)
        self.layer_lstm_word = nn.LSTM(self.word_embedding_dim,
                                        self.hidden_size_word,
                                        num_layers=cf.lstm_cnn_char_word['num_layer_lstm_word'],
                                        batch_first=True,
                                        bidirectional=False)
        
        self.reduce_size = cf.lstm_cnn_char_word['reduce_size']
        if self.reduce_size:
            self.fc = nn.Linear(self.hidden_size_char+self.hidden_size_word, cf.lstm_cnn_char_word['hidden_layer_fc'])
            self.label = nn.Linear(cf.lstm_cnn_char_word['hidden_layer_fc'], self.num_classes)
        else:
            self.label = nn.Linear(self.hidden_size_word + self.hidden_size_char, self.num_classes) 

    def attention_net(self, network_output, final_state):
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(network_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(network_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state

    def compute_forward(self, batch): 
        inputs_word_emb = self.word_embedding_layer(batch.word)
        output_hidden_word, (final_hidden_state_word, final_cell_state_word) = self.layer_lstm_word(inputs_word_emb) 
        attn_output = self.attention_net(output_hidden_word, final_hidden_state_word) 
        word_ft = attn_output.squeeze(0) 

        if self.char_embedding_dim != 0:
            inputs_char_emb = self.char_embedding_layer(batch.char)       
            char_ft = self.layer_char_cnn(inputs_char_emb)      
            ft = torch.cat([word_ft, char_ft],-1)
            if self.reduce_size:
                ft = self.fc(ft)
        return ft 

    def forward(self, batch):
        with torch.no_grad():
            output_ft = self.compute_forward(batch)
            output_predictions = self.label(output_ft)
        return output_predictions

    def loss(self, batch):
        labels = batch.label
        output_ft = self.compute_forward(batch)
        output_predictions = self.label(output_ft)
        log_probs = F.log_softmax(output_predictions)
        loss = F.nll_loss(log_probs.view(-1, self.num_classes), labels.view(-1))
        return loss 

    def get_accuracy(self, data_iter):
        correct, total = 0, 0
        for i, batch in enumerate(data_iter):
            output = self.forward(batch) 
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(batch.label.view_as(pred)).sum().item()
            total += batch.label.shape[0]
        return correct / total