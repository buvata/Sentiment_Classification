import torch
import string
import torch.nn as nn 
import torch.nn.functional as F 
import torchtext
from torchtext import vocab
from torchtext import data 
import config as cf   


def word_tokenize(x):
    word_tokenize = lambda x: x.split(" ")
    return word_tokenize

def get_dataloader_word(cf):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    txt_field = data.Field(sequential=True, 
                        tokenize=lambda x:x.split(" "), 
                        #include_lengths=True, 
                        use_vocab=True)

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
        ('combine_comment', txt_field) # process it as text
    ]

    trainds, valds = data.TabularDataset.splits(path = cf.data['path'], 
                                                format = cf.data['format'], 
                                                train = cf.data['train'], 
                                                validation = cf.data['validate'],
                                                fields = train_val_fields, 
                                                skip_header = True)

    train_iter, vald_iter = data.BucketIterator.splits(datasets = (trainds, valds), # train and validation Tabulardataset
                                                batch_sizes = (cf.data['batchsize'],cf.data['batchsize']),  # batch size of train and validation
                                                sort_key = lambda x: len(x.combine_comment),
                                                repeat = False,
                                                device = device)

    return train_iter, vald_iter, trainds, valds, txt_field, label_field


def get_dataloader_char(cf):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    txt_field = data.Field(sequential=True, 
                        tokenize=lambda x:x, 
                        #include_lengths=True, 
                        fix_length = 1014, 
                        use_vocab=True)

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
    ('combine_comment', txt_field) # process it as text
]

    trainds, valds = data.TabularDataset.splits(path = cf.data['path'], 
                                            format = cf.data['format'], 
                                            train = cf.data['train'], 
                                            validation = cf.data['validate'],
                                            fields = train_val_fields, 
                                            skip_header = True)

    train_iter, vald_iter = data.BucketIterator.splits(datasets = (trainds, valds), # train and validation Tabulardataset
                                            batch_sizes = (1,1),  # batch size of train and validation
                                            sort_key = lambda x: len(x.combine_comment),
                                            repeat = False,
                                            device = device)
    

    return train_iter, vald_iter, trainds, valds, txt_field, label_field



def get_dataloader_word_char(cf):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    txt_field = data.Field(sequential = True, 
                    tokenize = lambda x : x.split(" "), 
                    #include_lengths=True, 
                    batch_first = True,
                    use_vocab = True,
                    init_token = "<SOS>", 
                    eos_token = "<EOS>",
                    pad_token = '<pad>',
                    )

    label_field = data.Field(sequential = False, 
                        use_vocab = False,     
                        is_target = True,      
                        batch_first = True,
                        unk_token = None,
                    )

    CHAR_NESTING = data.Field(tokenize=list, init_token="<SOS>", eos_token="<EOS>", pad_token="<pad>", 
                            batch_first=True, fix_length=cf.lstm_char_word['fix_length_char'])
                            
    char_field = data.NestedField(CHAR_NESTING, init_token="<SOS>", eos_token="<EOS>", pad_token="<pad>")

    fields = [('id', None), 
        ('comment', None),
        ('label', label_field),
        ('comment_w_tone', None),
        (('word', 'char'), 
        (txt_field, char_field))]


    trainds, valds = data.TabularDataset.splits(path = cf.data['path'], 
                                            format = cf.data['format'], 
                                            train = cf.data['train'], 
                                            validation = cf.data['validate'],
                                            fields = fields, 
                                            skip_header = True)

    train_iter, vald_iter = data.BucketIterator.splits( datasets = (trainds, valds), # train  Tabulardataset
                                                    batch_sizes = (cf.data['batchsize'],cf.data['batchsize']),  # batch size of train
                                                    sort_key = lambda x : len(x.word),
                                                    sort_within_batch = True,
                                                    repeat = False,
                                                    device = device)


    return train_iter, vald_iter, trainds, valds, txt_field, char_field



