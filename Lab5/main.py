import tensorflow as tf

from model import InceptionV3


def train_model(train_ds, validation_ds, test_ds, epochs, batch_size):
    model = InceptionV3()

    initial_learning_rate = 10 ** (-3)
    final_learning_rate = 10 ** (-7)
    learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (1 / epochs)
    steps_per_epoch = len(train_ds)

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=steps_per_epoch,
        decay_rate=learning_rate_decay_factor
    )

    model.compile(loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate))

    model.summary()

    model.fit(
        train_ds,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_ds,
        verbose=1)

    model.evaluate(test_ds)

    return model


if __name__ == '__main__':
    print('Hello')
