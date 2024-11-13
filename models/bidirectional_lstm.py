import tensorflow as tf
import numpy as np
from gensim.models import KeyedVectors
from utils.constants import GOOGLE_W2V

class BidirectionalLSTM(tf.keras.Model):
    """
    A Bidirectional LSTM model for sequence prediction tasks.

    Attributes:
    - context_size: The size of the input context window.
    - vocab_size: The size of the vocabulary.
    - embedding_dim: The dimension of the word embeddings.
    - word2idx: A dictionary mapping words to their respective indices.
    - hidden_units: The number of units in the LSTM layer.
    - n_rnn_layers: The number of recurrent layers.
    - dropout_rate: The dropout rate for regularization.
    - learning_rate: The learning rate for the optimizer.
    """
    def __init__(self, context_size, vocab_size, embedding_dim, word2idx, hidden_units=128,
                 n_rnn_layers=1, dropout_rate=0.2, learning_rate=1e-3):
        super(BidirectionalLSTM, self).__init__()
        self.context_size = context_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.word2idx = word2idx
        self.hidden_units = hidden_units
        self.n_rnn_layers = n_rnn_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        # Build model layers
        self.embedding = self._build_embedding_layer()
        self.bi_lstm = self._build_bi_lstm_layer()
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.fc_layer = tf.keras.layers.Dense(self.vocab_size, activation='softmax')

    def build(self, input_shape):
        """Build the model with the given input shape."""
        super(BidirectionalLSTM, self).build(input_shape)

    def _build_embedding_layer(self):
        """
        Create an embedding layer using pretrained word vectors.

        Returns:
        - A Keras Embedding layer with pretrained weights.
        """
        embedding_matrix = self._load_pretrained_embeddings()
        return tf.keras.layers.Embedding(input_dim=self.vocab_size + 1,
                                         output_dim=self.embedding_dim,
                                         weights=[embedding_matrix],
                                         trainable=False)

    def _load_pretrained_embeddings(self):
        """
        Load pretrained word vectors from a file.

        Returns:
        - A numpy array representing the embedding matrix.
        """
        word2vec = KeyedVectors.load_word2vec_format(GOOGLE_W2V, binary=True)
        embedding_matrix = np.zeros((self.vocab_size + 1, self.embedding_dim))
        for word, i in self.word2idx.items():
            if word in word2vec:
                embedding_matrix[i] = word2vec[word]
        return embedding_matrix

    def _build_bi_lstm_layer(self):
        """
        Create a bidirectional LSTM layer.

        Returns:
        - A Keras Bidirectional LSTM layer.
        """
        return tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.hidden_units, return_sequences=False, dropout=self.dropout_rate))

    def call(self, inputs):
        """
        Define the forward pass of the model.

        Parameters:
        - inputs: The input data

        Returns:
        - The output of the model
        """
        x = self.embedding(inputs)
        x = self.bi_lstm(x)
        x = self.dropout(x)
        output = self.fc_layer(x)
        return output

    def compile_and_train(self, X_train, y_train, X_val, y_val, batch_size=64, epochs=10):
        """
        Compile and train the model.

        Parameters:
        - X_train: Training data features
        - y_train: Training data labels
        - X_val: Validation data features
        - y_val: Validation data labels
        - batch_size: Number of samples per batch
        - epochs: Number of training epochs

        Returns:
        - Training history
        """
        self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

        history = self.fit(X_train, y_train, validation_data=(X_val, y_val),
                           batch_size=batch_size, epochs=epochs,
                           callbacks=[early_stopping, lr_scheduler])
        return history

# # Example usage
# context_size = 100
# vocab_size = 10000
# embedding_dim = 300
# word2idx = {'example': 1}  # Example vocabulary
#
# model = BidirectionalLSTM(context_size, vocab_size, embedding_dim, word2idx)
# model.build(input_shape=(None, context_size))
# model.summary()
