import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from model import FeedForwardModel, CascadeForwardModel, ElmanModel
from function import data_split, generate_data, load_data


DATA_PATH = "Data\\data.csv"


def learning_rate():
    return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=100,
        decay_rate=0.99
    )


def learning_rate2(epochs, batch_size):
    initial_learning_rate = 10 ** (-3)
    final_learning_rate = 10 ** (-7)
    learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (1 / epochs)
    steps_per_epoch = int(len(train) / batch_size)

    return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=steps_per_epoch,
        decay_rate=learning_rate_decay_factor
    )


def train_model(model_name, hidden_neurons, train, val, test, epochs, batch_size, learning_rate):

    model = None
    if model_name == 'FeedForward':
        model = FeedForwardModel(2, hidden_neurons)
    elif model_name == 'CascadeForward':
        model = CascadeForwardModel(2, hidden_neurons)
    elif model_name == 'Elman':
        model = ElmanModel(2, hidden_neurons)

    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.SGD(
                      learning_rate=learning_rate))

    model.summary()

    log = model.fit(
        np.reshape(train[:, :2], (-1, 2)),
        train[:, 2],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(np.reshape(val[:, :2], (-1, 2)), val[:, 2]),
        verbose=1)

    return log


def graph(train_loss, val_loss):
    plt.title('Mean_squared_error')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(train_loss, label='Train loss')
    plt.plot(val_loss, label='Validation loss')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    #generate_data(DATA_PATH, 1000000)

    data = load_data(DATA_PATH)
    train, val, test = data_split(data, 0.7, 0.1, 0.2)

    epochs = 500
    batch_size = 10000

    lr = learning_rate()
    lr2 = learning_rate2(epochs, batch_size)

    log = train_model("FeedForward", [10], train, val, test, epochs, batch_size, lr2)
    graph(log.history['loss'], log.history['val_loss'])

    log = train_model("FeedForward", [20], train, val, test, epochs, batch_size, lr2)
    graph(log.history['loss'], log.history['val_loss'])

    log = train_model("CascadeForward", [20], train, val, test, epochs, batch_size, lr2)
    graph(log.history['loss'], log.history['val_loss'])

    log = train_model("CascadeForward", [10, 10], train, val, test, epochs, batch_size, lr)
    graph(log.history['loss'], log.history['val_loss'])

    log = train_model("Elman", [15], train, val, test, epochs, batch_size, lr2)
    graph(log.history['loss'], log.history['val_loss'])

    log = train_model("Elman", [10, 10, 10], train, val, test, epochs, batch_size, lr2)
    graph(log.history['loss'], log.history['val_loss'])


