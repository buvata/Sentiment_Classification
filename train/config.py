data={
    'path' : "../data/data_model/" ,
    'format' : "csv" ,
    'train' : 'data_train.csv' ,
    'validate' : 'data_validate.csv' ,
    'test' : '../data/data_model/data_test.csv',
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
    'modelfnamepath' : "../model_results/",
    'modelfname' : " ",
    'save_device' : 'cpu'  # or 'cuda' , 'cpu'
}

cnn_word = {
    'cnn_filter_num': 5,
    'window_size': [3,4],
    'embedding_dim' : 100, 
    'output_dim' : 1, 
}

lstm_cnn_char_word = {
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
    'reduce_size' : True,
    'hidden_layer_fc' : 8,
    'use_cnn_feature_char' : True ,
    'save_model' : False,
    'char_cnn_filter_num': 100,
    'char_window_size': [3,4,5],
    'dropout_cnn': 0.5,
    'dropout_all' : 0.5
}


model_train = {
    'learning_rate': 1e-4,
    'num_epochs' : 2,
    'dropout' : 0.5,
    'save_model' : True ,
    'save_device' :'cpu',
    'resultsdfpath' : "../model_results/result.p",
    'modelfnamepath' : "../model_results/",
    'modelfname' : " ",
    'batchsize' : 32,
    'mode' : 'lstm_cnn_word_char'   # 'lstm_word_char' , 'rnn_word' , 'cnn_word_char'
}

