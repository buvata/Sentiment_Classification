data={
    'path' : "../data" ,
    'format' : "csv" ,
    'train' : 'data_train.csv' ,
    'validate' : 'data_validate.csv' ,
    'test' : '../data/data_test.csv',
    'batchsize' : 32
}

lstm_char_word = {
    'num_classes' : 2 ,
    'char_embedding_dim' : 64 ,
    'word_embedding_dim' : 100 ,
    'hidden_size_char' : 0 ,
    'hidden_size_word' : 16 ,
    'num_layer_lstm_char' : 1 , 
    'num_layer_lstm_word' : 1 ,
    'num_epochs' : 2, 
    'learning_rate' : 1e-4, 
    'batch_size' : 32,
    'use_char_cnn' : True ,
    'save_model' : True,
    'use_highway_char' : True,
    'char_cnn_filter_num': 100,
    'char_window_size': [3,4,5],
    'dropout_cnn': 0.5,
    'fix_length_char': 6,
    'resultsdfpath' : "../model_results/result.p",
    'modelfname' : " ",
    'save_device' : 'cpu'  # or 'cuda' 
}

