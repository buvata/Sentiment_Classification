import config as cf  
import torch.nn as nn 
import os 
import sys 
sys.path.insert(0,"../")
from evaluate.evaluate import *
from utils import * 
from LSTM_char_word import LSTMWordChar


def train_lstm_network(cf, model, train, valid): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train.init_epoch()
    valid.init_epoch()

    if cf.lstm_char_word['save_device'] == 'cpu':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)

    for param in model.parameters():
        param.requires_grad = True 
    optimizer = torch.optim.Adam(model.parameters(), lr = cf.lstm_char_word['learning_rate'])
    losses, train_acc, valid_acc, loss = [], [], [], 0
    epochs = []
    best_valid_acc = 0.
    best_epoch = 0 
    for epoch in range(cf.lstm_char_word['num_epochs']):
        print(epoch)
        for i, batch in enumerate(train): 
            l = batch.label   
            optimizer.zero_grad()
            pred = model.compute_forward(batch)
            loss = criterion(pred, l)  
            loss.backward()  
            optimizer.step()
        
        losses.append(float(loss))

        epochs.append(epoch)
        train_acc.append(model.get_accuracy(train))
        valid_acc.append(model.get_accuracy(valid))

        if valid_acc[-1] > best_valid_acc:
            best_valid_acc = valid_acc[-1]
            best_epoch = epoch 
            if cf.lstm_char_word['save_model']:
                cf.lstm_char_word['modelfname'] = "model_cf" + \
                "_".join(["_use_cnn", str(cf.lstm_char_word['use_char_cnn'])]) + \
                "_".join(["_w_emb", str(cf.lstm_char_word['word_embedding_dim'])]) + \
                "_".join(["_lr", str(cf.lstm_char_word['learning_rate'])]) + \
                "_".join(["_bs", str(cf.lstm_char_word['batch_size'])]) + \
                "_".join(["_hidden_size_w", str(cf.lstm_char_word['hidden_size_word'])]) + \
                ".pt"
                model.save(cf.lstm_char_word['modelfname'])
        print("Epoch %d; Loss %f; Train Acc %f ; Val Acc %f " % (
            epoch+1, loss, train_acc[-1], valid_acc[-1]))

    if cf.lstm_char_word['save_model']:
        if not os.path.isfile(cf.lstm_char_word['resultsdfpath']):
            results_df = pd.DataFrame(columns = ["modelname", "best_acc", "best_epoch"])
            experiment_df = pd.DataFrame([[cf.lstm_char_word['modelfname'], best_valid_acc, best_epoch]],
                columns = ["modelname", "best_acc", "best_epoch"])
            results_df = results_df.append(experiment_df, ignore_index=True)
            results_df.to_pickle(cf.lstm_char_word['resultsdfpath'])
        else:
            results_df = pd.read_pickle(cf.lstm_char_word['resultsdfpath'])
            experiment_df = pd.DataFrame([[cf.lstm_char_word['modelfname'], best_valid_acc, best_epoch]],
                columns = ["modelname", "best_acc", "best_epoch"])
            results_df = results_df.append(experiment_df, ignore_index=True)
            results_df.to_pickle(cf.lstm_char_word['resultsdfpath'])

if __name__ == "__main__":
   
    train_iter, vald_iter, trainds, valds, txt_field, char_field = get_dataloader_word_char(cf)
    build_vocab(txt_field, trainds)
    build_vocab(char_field, trainds)

    vocab_word = txt_field.vocab
    vocab_char = char_field.vocab

    model = LSTMWordChar(vocab_word, vocab_char, cf )
  
    train_mode = True
    if train_mode is True :
        train_lstm_network(cf, model, train_iter, vald_iter ) 
    else:
        model_test = load_model(cf,model,'model_cf_use_cnn_True_w_emb_100_lr_0.0001_bs_32_hidden_size_w_16.pt')
        plot_confusion_matrix(model_test,cf) 








