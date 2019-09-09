from __future__ import absolute_import

import torch
import string
import torch.nn as nn 
import torch.nn.functional as F 
import torchtext
from torchtext import vocab
from torchtext import data 
import config as cf   
from data.dataloader import *


def load_model(cf,model, path_save_model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')     
    # save GPU, load CPU 
    if device == torch.device('cpu') and cf.lstm_char_word['save_device'] == 'cuda': 
        model.load_state_dict(torch.load(path_save_model),map_location=device)
    # save GPU, load GPU 
    if device == torch.device('cuda') and cf.lstm_char_word['save_device'] == 'cuda':
        model.load_state_dict(torch.load(path_save_model))
        model = model.to(device)
    # save CPU, load CPU 
    if device == torch.device('cpu') and cf.lstm_char_word['save_device'] == 'cpu':
        model.load_state_dict(torch.load(path_save_model))
    return model
   

def build_vocab(data_field, data_iter):
    vocab = data_field.build_vocab(data_iter)
    return vocab 


def load_vocab(cf):
    train_iter, vald_iter, trainds, valds, txt_field, char_field = get_dataloader_word_char(cf)

    build_vocab(txt_field, trainds)
    build_vocab(char_field, trainds)

    vocab_word = txt_field.vocab
    vocab_char = char_field.vocab

    return vocab_word, vocab_char 