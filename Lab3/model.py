import tensorflow as tf


def model(neurons_in_hidden_layers):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(28, 28)))
    model.add(tf.keras.layers.Flatten())

    for neurons in neurons_in_hidden_layers:
        model.add(tf.keras.layers.Dense(neurons, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    return model