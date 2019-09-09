import numpy as np
import pandas as pd
import math
import os
import argparse
import torchtext
import heapq
import torch
import time
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import config as cf 
from utils import * 
from sub_layer.cnn_based_char import CharCNN
from sub_layer.highway import Highway


class LSTMWordChar(nn.Module):
    def __init__(self, vocab_word, vocab_char, cf):
        
        super(LSTMWordChar, self).__init__()

        self.word_embedding_dim = cf.lstm_char_word['word_embedding_dim']
        self.char_embedding_dim = cf.lstm_char_word['char_embedding_dim']
        self.hidden_size_word = cf.lstm_char_word['hidden_size_word']
        self.num_classes = cf.lstm_char_word['num_classes']

        self.vocab_word = vocab_word
        self.vocab_char = vocab_char
        self.word_embedding_layer = nn.Embedding(len(vocab_word), self.word_embedding_dim)
        self.char_embedding_layer = None

        self.char_embedding_dim = cf.lstm_char_word['char_embedding_dim'] 
        self.hidden_size_char = cf.lstm_char_word['hidden_size_char']   
        
        self.use_highway_char = cf.lstm_char_word['use_highway_char']
        self.use_char_cnn = cf.lstm_char_word['use_char_cnn']
        if self.use_char_cnn:
            self.char_cnn_filter_num = cf.lstm_char_word['char_cnn_filter_num']
            self.char_window_size = cf.lstm_char_word['char_window_size']
            self.dropout_cnn = cf.lstm_char_word['dropout_cnn']
            self.fix_length_char = cf.lstm_char_word['fix_length_char']
        

        if vocab_char is not None and self.char_embedding_dim > 0:
            self.char_embedding_layer = nn.Embedding(len(vocab_char), self.char_embedding_dim)

            if not self.use_char_cnn:
                self.lstm_char = nn.LSTM(self.char_embedding_dim,
                                        self.hidden_size_char,
                                        num_layers=cf.lstm_char_word['num_layer_lstm_char'],
                                        batch_first=True,
                                        bidirectional=False)

                if self.use_highway_char:
                    self.highway_char = Highway(self.hidden_size_char , num_layers=1, f=torch.relu)

            else:
                self.layer_char_cnn = CharCNN(self.char_embedding_dim,
                                              self.char_cnn_filter_num,
                                              self.char_window_size,
                                              self.dropout_cnn)
               
        if self.char_embedding_dim == 0:
            self.embedding_word_lstm = self.word_embedding_dim
        else:
            if not self.use_char_cnn:
                self.embedding_word_lstm = self.hidden_size_char + self.word_embedding_dim
            else: 
                each_number_output_filter = self.char_cnn_filter_num * len(self.char_window_size) 
                self.hidden_size_char = each_number_output_filter
                if self.use_highway_char:
                    self.highway_char = Highway(self.hidden_size_char , num_layers=1, f=torch.relu)

                self.embedding_word_lstm = self.hidden_size_char + self.word_embedding_dim
   
        '''       
        if vocab_word.vectors is not None:
            if word_embedding_dim != vocab_word.vectors.shape[1]:
                raise ValueError("expect embedding word: {} but got {}".format(word_embedding_dim,
                                                                               vocab_word.vectors.shape[1]))

            self.word_embedding_layer.weight.data.copy_(vocab_word.vectors)
            self.word_embedding_layer.requires_grad = False
        '''

        # lstm get input shape (seq_len, batch_size, input_dim)
        self.lstm_word = nn.LSTM(self.embedding_word_lstm,
                                 self.hidden_size_word,
                                   num_layers=cf.lstm_char_word['num_layer_lstm_word'],
                                   batch_first=True,
                                   bidirectional=False)
        self.label = nn.Linear(self.hidden_size_word, self.num_classes) 

    def attention_net(self, lstm_output, lstm_final_state):
        hidden = lstm_final_state.squeeze(0)
    
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
     
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state
  
    def compute_forward(self, batch):

        # input_word_emb = [batch_size, seq_sent, word_emb_dim]

        inputs_word_emb = self.word_embedding_layer(batch.word)    

        if self.char_embedding_layer is not None:

            # batch.char = [batch_size, seq_len, max_len_word]
            # input_char_emb = [batch x seq_len, max_len_word, char_emb_dim]
            inputs_char_emb = self.char_embedding_layer(batch.char.view(-1, batch.char.shape[-1])) 
         
            if not self.use_char_cnn:
                seq_len = inputs_word_emb.shape[1]

                # final_hidden_state_char = [1, batch x seq_len, hidden_size_char]
                _, (final_hidden_state_char, _) = self.lstm_char(inputs_char_emb)

                # input_char_emb = [batch, seq_len, hidden_size_char]    
                inputs_char_emb = final_hidden_state_char.view(-1, seq_len, self.hidden_size_char)

                if self.use_highway_char:
                    final_hidden_state_char = self.highway_char(inputs_char_emb)    
            else:   
                each_word_conv_output = self.layer_char_cnn(inputs_char_emb)
                if self.use_highway_char:
                    each_word_conv_output = self.highway_char(each_word_conv_output)

                inputs_char_emb = each_word_conv_output.view(batch.char.shape[0],
                                                            batch.char.shape[1],
                                                            each_word_conv_output.shape[1])
              
            
            inputs_word_emb = torch.cat([inputs_word_emb, inputs_char_emb], -1)
          
        # output_hidden_word = [batch, seq_len, hidden_size_word]
        # final_hidden_state = [1, batch, hidden_size_word]
        output_hidden_word, (final_hidden_state_word, final_cell_state_word) = self.lstm_word(inputs_word_emb)
       
        attn_output = self.attention_net(output_hidden_word, final_hidden_state_word)

        logits = self.label(attn_output)
      
        return logits


    def loss(self, batch):
        logits_predict = self.compute_forward(batch)
        predict_value = torch.max(logits_predict, 1)[1]
        target = batch.label
        loss = F.cross_entropy(logits_predict, target)
        return loss


    def get_accuracy(self, data_iter):
        correct, total = 0, 0
        for i, batch in enumerate(data_iter):
            output = self.forward(batch) 
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(batch.label.view_as(pred)).sum().item()
            total += batch.label.shape[0]
        return correct / total


    def forward(self, batch):
        with torch.no_grad():
            logits = self.compute_forward(batch)
        return logits

   
    def save(self, path_save_model):
        torch.save(self.state_dict(), path_save_model)

