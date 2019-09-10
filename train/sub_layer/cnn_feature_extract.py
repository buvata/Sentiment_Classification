import torch
import torch.nn as nn
from torch.nn import functional as F


class CNNFeatureExtract(nn.Module):
    def __init__(self, embedding_dim, filter_num,
                 kernel_sizes=[2, 3],
                 dropout_cnn=0.5):

        super(CNNFeatureExtract, self).__init__()
        self.embedding_dim = embedding_dim
        self.filter_num = filter_num
        self.kernel_sizes = kernel_sizes

        self.convs = nn.ModuleList([nn.Conv2d(1,
                                              self.filter_num,
                                              (kernel_size, self.embedding_dim)) for kernel_size in self.kernel_sizes])

        self.dropout_layer = nn.Dropout(dropout_cnn)

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)
        output_relu = F.relu(conv_out).squeeze(3)
        output_maxpool1d = F.max_pool1d(output_relu, output_relu.size(2)).squeeze(2)
        return output_maxpool1d

    def forward(self, embedding_feature):
        x = embedding_feature.unsqueeze(1)
        # embedding feature has shape (batch_size, num_seq, embedding_length)
        list_output_cnn = [self.dropout_layer(self.conv_block(x, conv_layer)) for conv_layer in self.convs]
        output_feature = torch.cat(list_output_cnn, 1)
        return output_feature


if __name__ == '__main__':
    x_test = torch.rand((2, 15, 64))
    cnn_layer = CNNFeatureExtract(64, 15)
    cnn_layer(x_test)