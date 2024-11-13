import tensorflow as tf
import tensorflow_federated as tff
import matplotlib.pyplot as plt
from utils.data_processors import extend_data


def preprocess_dataset(dataset):
    """
    Preprocess the dataset by batching and formatting.

    Parameters:
    - dataset: The input dataset

    Returns:
    - Preprocessed dataset
    """

    def batch_format_function(element):
        return (tf.reshape(element["x"], [-1, 5]), tf.reshape(element["y"], [-1, 1]))

    return dataset.batch(20).map(batch_format_function)


def create_federated_model():
    """
    Create a TFF model from a Keras model.

    Returns:
    - TFF model
    """
    keras_model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(10, activation="relu", input_shape=(5,)),
            tf.keras.layers.Dense(1, activation="softmax"),
        ]
    )
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=preprocess_dataset(
            tff.simulation.datasets.emnist.load_data()[0]
        ).element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )


def prepare_federated_data(data, user_list, context_size):
    """
    Prepare federated data for each user.

    Parameters:
    - data: The dataset
    - user_list: List of users
    - context_size: Context window size

    Returns:
    - List of federated datasets
    """
    federated_datasets = []
    for user in user_list:
        user_data = data[data.user == user]
        X, Y = extend_data(user_data.X, user_data.Y, context_size)
        federated_datasets.append(
            preprocess_dataset(tf.data.Dataset.from_tensor_slices({"x": X, "y": Y}))
        )
    return federated_datasets


def train_federated_model(federated_model, federated_datasets, num_rounds=10):
    """
    Train the federated model using the federated averaging process.

    Parameters:
    - federated_model: The TFF model
    - federated_datasets: List of federated datasets
    - num_rounds: Number of training rounds

    Returns:
    - Final state of the model
    """
    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn=create_federated_model,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    )

    state = iterative_process.initialize()

    for round_number in range(num_rounds):
        state, metrics = iterative_process.next(state, federated_datasets)
        print(f"Round {round_number + 1}, Metrics={metrics}")
    return state


def plot_training_metrics(training_history):
    """
    Plot the training metrics.

    Parameters:
    - training_history: Dictionary containing training metrics
    """
    plt.plot(training_history["sparse_categorical_accuracy"], label="Accuracy")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Federated Learning Metrics")
    plt.legend()
    plt.show()
