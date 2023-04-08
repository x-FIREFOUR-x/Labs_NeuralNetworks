import tensorflow as tf


def FeedForwardModel(input_size=2, neurons_in_hidden_layers=[10]):
    inputs = tf.keras.layers.Input(input_size)
    layer = inputs
    for neurons in neurons_in_hidden_layers:
        layer = tf.keras.layers.Dense(neurons, activation='relu')(layer)
    outputs = tf.keras.layers.Dense(1, activation='relu')(layer)

    model = tf.keras.Model(inputs, outputs)
    return model


def CascadeForwardModel(input_size=2, neurons_in_hidden_layers=[10]):
    inputs = tf.keras.layers.Input(input_size)
    concat = inputs
    for neurons in neurons_in_hidden_layers:
        hidden_layer = tf.keras.layers.Dense(neurons, activation='relu')(concat)
        concat = tf.keras.layers.Concatenate(axis=-1)([concat, hidden_layer])
    outputs = tf.keras.layers.Dense(1, activation='relu')(concat)

    model = tf.keras.Model(inputs, outputs)
    return model


def ElmanModel(input_size=2, hidden_neurons=[10]):
    inputs = tf.keras.layers.Input(input_size)

    layer = tf.expand_dims(inputs, axis=1)
    layer = tf.keras.layers.SimpleRNN(hidden_neurons[0])(layer)
    for neurons in hidden_neurons[1:]:
        layer = tf.expand_dims(layer, axis=1)
        layer = tf.keras.layers.SimpleRNN(neurons, activation='relu')(layer)
    outputs = tf.keras.layers.Dense(1, activation='relu')(layer)

    model = tf.keras.Model(inputs, outputs)
    return model
