import numpy as np
import pandas as pd
import math
import os
import argparse
import torchtext
from torchtext import data
from torchtext.data import Iterator, BucketIterator
import spacy
import heapq
import torch
import time
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader, Dataset 
from torchtext import vocab
from torchtext.vocab import SubwordVocab
import matplotlib.pyplot as plt 


# define the columns that we want to process 
txt_field = data.Field(sequential=True, 
                       tokenize=lambda x:x.split(), 
                       include_lengths=True, 
                       batch_first=True,
                       use_vocab=True
                      )

label_field = data.Field(sequential=False, 
                         use_vocab=False,     
                         is_target=True,      
                         batch_first=True,
                         unk_token=None)


train_val_fields = [
    ('id', None),
    ('comment',None),
    ('label', label_field), # process it as label
    ('comment_w_tone', None),
    ('combine_comment', txt_field), # process it as text   
]


trainds, valds = data.TabularDataset.splits(path = '../data', 
                                            format = 'csv', 
                                            train = 'data_train.csv', 
                                            validation = 'data_validate.csv',
                                            fields = train_val_fields, 
                                            skip_header = True)

traindl, valdl = data.BucketIterator.splits(datasets = (trainds, valds), # train  Tabulardataset
                                            batch_sizes = (32,32),  # batch size of train
                                            sort_key = lambda x: len(x.combine_comment),
                                            sort_within_batch = True,
                                            repeat = False)


class BatchGenerator:
    def __init__(self, dl, x_field, y_field):
        self.dl, self.x_field, self.y_field = dl, x_field, y_field
        
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        for batch in self.dl:
            X = getattr(batch, self.x_field)
            y = getattr(batch, self.y_field)
            yield (X,y)
            
#train_batch_it = BatchGenerator(traindl, 'combine_comment', 'label')
#val_batch_it = BatchGenerator(valdl, 'combine_comment', 'label')

txt_field.build_vocab(trainds, max_size=10000)
label_field.build_vocab(trainds)


class RNN_Text_Classification(nn.Module):
    def __init__(self, input_size, vocab_size, hidden_size, num_classes):      
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.hidden_size = hidden_size
        self.rnn_layer = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        #text = [seq len, batch size]
        #embedded = [seq len, batch size, emb size]
        x = self.embedding(x)
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        #output = [seq len, batch size, hid size]
        #hidden = [1, batch size, hid size]
        out, _ = self.rnn_layer(x, h0)
        out = self.fc(torch.max(out, dim=1)[0])
        return out


def get_accuracy(model, data):
    data_iter = torchtext.data.BucketIterator(data, 
                                              batch_size=64, 
                                              sort_key=lambda x: len(x.combine_comment), 
                                              repeat=False)
    correct, total = 0, 0
    for i, batch in enumerate(data_iter):
        output = model(batch.combine_comment[0]) 
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(batch.label.view_as(pred)).sum().item()
        total += batch.combine_comment[1].shape[0]
    return correct / total

def train_rnn_network(model, train, valid, num_epochs=5, learning_rate=1e-5, batch_size=32):
    train_iter = torchtext.data.BucketIterator(train,
                                           batch_size=batch_size,
                                           sort_key=lambda x: len(x.combine_comment), # to minimize padding
                                           sort_within_batch=True,        # sort within each batch
                                           repeat=False)                
    
    for param in model.parameters():
        param.requires_grad = True
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses, train_acc, valid_acc, loss = [], [], [], 0
    epochs = []
    for epoch in range(num_epochs):
        print(epoch)
        for j, batch in enumerate(train_iter):
            l = batch.label
            s = batch.combine_comment
        
            optimizer.zero_grad()
            pred = model(s[0])
            loss = criterion(pred, l)
            
            loss.backward()
            optimizer.step()
    
        losses.append(float(loss))

        epochs.append(epoch)
        train_acc.append(get_accuracy(model, train))
        valid_acc.append(get_accuracy(model, valid))
        print("Epoch %d; Loss %f; Train Acc %f; Val Acc %f" % (
            epoch+1, loss, train_acc[-1], valid_acc[-1]))
    

if __name__ == "__main__":
    Input_size = 128
    Vocab_size = len(txt_field.vocab.stoi)
    Hidden_size = 128
    Num_classes = 2
    model = RNN_Text_Classification(Input_size, Vocab_size, Hidden_size, Num_classes)

    train_rnn_network(model, trainds, valds, num_epochs=5, learning_rate=1e-5)
