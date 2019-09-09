import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from utils import *
import config as cf 


class CharCNN(nn.Module):
    def __init__(self, char_embedding_dim, filter_num, windows_size,
                 dropout_cnn):
        super(CharCNN, self).__init__()

        self.char_embedding_dim = char_embedding_dim
        self.char_cnn_filter_num = filter_num
        self.windows_size = windows_size

        self.conv1ds = nn.ModuleList([nn.Conv1d(in_channels=self.char_embedding_dim,
                                                out_channels=self.char_cnn_filter_num,
                                                kernel_size=k) for k in windows_size
                                        ])

        self.dropout_cnn = nn.Dropout(dropout_cnn)

    def forward(self, char_embedding_feature):
    
        #  char_embedding_feature: shape: (batch_size, max_len character of word, embedding dim) 
        char_embedding_feature = char_embedding_feature.permute(0, 2, 1)

        conved = [F.relu(conv(char_embedding_feature)) for conv in self.conv1ds]
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
  
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
       
        cat = self.dropout_cnn(torch.cat(pooled, dim = 1))
      
        return cat

