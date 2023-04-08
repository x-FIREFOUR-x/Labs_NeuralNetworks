import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys

from model import FeedForwardModel, CascadeForwardModel, ElmanModel
from function import data_split, generate_data, load_data

DATA_PATH = "Data\\data100.csv"


def train_model(model_name, hidden_neurons, train, val, epochs, batch_size):

    initial_learning_rate = 10 ** (-3)
    final_learning_rate = 10 ** (-7)
    learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (1 / epochs)
    steps_per_epoch = int(len(train) / batch_size)
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=steps_per_epoch,
        decay_rate=learning_rate_decay_factor
    )

    if model_name == 'FeedForward':
        model = FeedForwardModel(2, hidden_neurons)
    elif model_name == 'CascadeForward':
        model = CascadeForwardModel(2, hidden_neurons)
    elif model_name == 'Elman':
        model = ElmanModel(2, hidden_neurons)

    model.compile(loss='mean_squared_error', metrics=['mean_absolute_error'],
                  optimizer=tf.keras.optimizers.SGD(
                      learning_rate=learning_rate))

    log = model.fit(
        np.reshape(train[:, :2], (-1, 2)),
        train[:, 2],
        batch_size,
        epochs=epochs,
        validation_data=(np.reshape(val[:, :2], (-1, 2)), val[:, 2]),
        verbose=1)

    return log


if __name__ == '__main__':
    # generate_data(DATA_PATH, 100)

    data = load_data(DATA_PATH)
    train, val, test = data_split(data, 50, 25, 25)

    log = train_model("FeedForward", [10], train, val, 100, 100)
    log = train_model("CascadeForward", [10], train, val, 100, 100)
    log = train_model("Elman", [10], train, val, 100, 100)


