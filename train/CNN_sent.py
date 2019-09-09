import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from torchtext import vocab
from torchtext import data 
from torchtext.data import Iterator, BucketIterator
import config as cf 


class CNN1d_word(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.convs = nn.ModuleList([
                                    nn.Conv1d(in_channels = embedding_dim, 
                                              out_channels = n_filters, 
                                              kernel_size = fs)
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        


    def forward(self, text):
        
        #text = [sent len, batch size]
        
        text = text.permute(1, 0)
      
        #text = [batch size, sent len]
        
        embedded = self.embedding(text)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.permute(0, 2, 1)
       
        #embedded = [batch size, emb dim, sent len]
        
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size, n_filters]
      
        
        cat = self.dropout(torch.cat(pooled, dim = 1))
       
        #cat = [batch size, n_filters * len(filter_sizes)]
        out = self.fc(cat)
        # out = out.squeeze()    
        # print("output shape:", cat.shape)
        return out


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc

def train(model, data_iter, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0    
 
    for batch in data_iter:   
        l = batch.label.float()
        s = batch.combine_comment

        optimizer.zero_grad()
        
        predictions = model(s).squeeze(1)
        
        loss = criterion(predictions, l)
        
        acc = binary_accuracy(predictions, l)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(data_iter), epoch_acc / len(data_iter)

def evaluate(model, data_iter, criterion):   
    epoch_loss = 0
    epoch_acc = 0   
    
    with torch.no_grad():

        for batch in data_iter:
            l = batch.label.float()
            s = batch.combine_comment

            predictions = model(s).squeeze(1)
            
            loss = criterion(predictions, l)
            
            acc = binary_accuracy(predictions, l)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(data_iter), epoch_acc / len(data_iter)


def train_model_CNN_word(model, train_iter, valid_iter, num_epochs):
  
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    
    train_iter.init_epoch()
    valid_iter.init_epoch()
    for epoch in range(num_epochs): 
        train_loss, train_acc = train(model, train_iter, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iter, criterion)
        '''
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'model.pt')
        '''
        print(f'Epoch: {epoch+1: 02} ')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
 

if __name__ == "__main__":

    traindl, valdl, trainds, valds, txt_field, txt_label = get_dataloader_word(cf)

    build_vocab(txt_field, trainds, None )
  
    input_dim = len(txt_field.vocab)
    embedding_dim = 100
    n_filters = 100
    filter_size = [2,3]
    output_size = 1
    dropout = 0.5
    pad_idx = txt_field.vocab.stoi[txt_field.pad_token]
    num_epochs = 5

    model = CNN1d_word(input_dim, embedding_dim, n_filters, filter_size, output_size, dropout, pad_idx)

    train_model_CNN_word(model, traindl, valdl, num_epochs)
    # batch = next(iter(traindl))
    # print(model(batch.combine_comment).squeeze(1))
    # print(batch.label)