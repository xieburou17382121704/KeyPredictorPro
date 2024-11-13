import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils.data_processors import extend_data


def train_local_model(
    model_class,
    word2idx,
    D,
    hidden_nodes,
    data,
    model_file,
    batch_size=256,
    context_size=5,
    epochs=10,
    validation_split=0.2,
    retrain=False,
    show_progress=True,
):
    # Check if model file exists and retrain is False

    if os.path.isfile(model_file) and not retrain:
        model = tf.keras.models.load_model(model_file)
        return model
    # Initialize model

    model = model_class(context_size, len(word2idx), D, word2idx, hidden_nodes)

    # Extend data

    X, Y = extend_data(data.X, data.Y, context_size)

    # Split data into training and validation sets

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=validation_split)

    # Compile model

    model.compile(
        optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Train model

    history = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, Y_val),
        verbose=1 if show_progress else 0,
    )

    # Save model

    model.save(model_file, save_format="tf")

    # Plot training history if required

    if show_progress:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"], label="train accuracy")
        plt.plot(history.history["val_accuracy"], label="val accuracy")
        plt.legend()
        plt.title("Accuracy")

        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"], label="train loss")
        plt.plot(history.history["val_loss"], label="val loss")
        plt.legend()
        plt.title("Loss")

        plt.show()
    return model
