import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from model import Model


def train_model(hidden_neurons, x_train, y_train, x_test, y_test, epochs, batch_size):
    model = Model(hidden_neurons)

    initial_learning_rate = 10 ** (-3)
    final_learning_rate = 10 ** (-7)
    learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (1 / epochs)
    steps_per_epoch = int(len(x_train) / batch_size)

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=steps_per_epoch,
        decay_rate=learning_rate_decay_factor
    )

    model.compile(loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    model.summary()

    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        verbose=1)

    return model


def predict_model(model, x, y):
    plot = plt.imshow(x)
    plt.show()
    predictions = model.predict(np.expand_dims(x, axis=0), verbose=0)
    print('Correct: ', y)
    print('Predicted: ', predictions.argmax())


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    model = train_model([75, 75, 75], x_train, y_train, x_test, y_test, 20, 100)
    model.evaluate(x_test, y_test)
    predict_model(model, x_test[0], y_test[0])

