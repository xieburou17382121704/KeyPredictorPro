from federated import federated_training
from local import local_training
from utils.data_processors import split_local_remote_data, merge_and_index_data
from models import bidirectional_lstm



if __name__ == "__main__":
    data_file = "data/training.1600000.processed.noemoticon.csv"
    model_file = "data/LSTM_model_local.h5"
    federated_file = "data/LSTM_model_federated.h5"
    dump_file = "data/pandas_df.pkl"
    word2idx_file = "data/tokenizer_keys.pkl"
    min_tweets = 20
    local_share = 0.2
    context_size = 5
    epochs = 1
    D = 300
    n_nodes = 128

    # Data processing

    data, word2idx = merge_and_index_data(
        data_file, dump_file, word2idx_file, min_tweets
    )
    local_data, remote_data = split_local_remote_data(data, local_share)

    # Model definition

    model = bidirectional_lstm.BidirectionalLSTM(
        context_size=context_size,
        vocab_size=len(word2idx),
        embedding_dim=D,
        word2idx=word2idx,
        hidden_units=n_nodes,  # Use hidden_units instead of n_nodes
    )
    # Compile the model

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Local training

    local_training.train_local_model(
        model_class=bidirectional_lstm.BidirectionalLSTM,  # Assume the model class needs to be passed
        data=local_data,
        model_file=model_file,
        word2idx=word2idx,
        D=D,
        hidden_nodes=n_nodes,
        epochs=epochs,
    )

    # Federated training

    federated_training.train_federated_model(model, remote_data, federated_file)
