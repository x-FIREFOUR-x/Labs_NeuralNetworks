import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

from processData import get_data
from model import AlexNet_model


CLASS_NAMES = ['airplane', 'automobile', 'bird', 'deer', 'cat', 'dog', 'frog', 'horse', 'ship', 'truck']


def predict_image(model, test_data, count):
    plt.figure(figsize=(4, 4))
    for (images,  correct_classes) in test_data.take(1):
        correct_classes = correct_classes.numpy().squeeze(axis=1)
        predictions = model.predict(images)
        predicted_classes = np.argmax(predictions, axis=1)

        for index, img in enumerate(images):
            plt.imshow(test_images[index])
            plt.show()
            print('Correct =', CLASS_NAMES[correct_classes[index]])
            print('Predicted =', CLASS_NAMES[predicted_classes[index]])

            if (index > count):
                break


def train_model(train_ds, validation_ds, test_ds, epochs, batch_size):
    model = AlexNet_model()

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



    model.fit(
        train_ds,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_ds,
        verbose=1)

    model.evaluate(test_ds)
    model.summary()

    return model


if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

    train_ds, validation_ds, test_ds = get_data(train_images, train_labels, test_images, test_labels)
    model = train_model(train_ds, validation_ds, test_ds, 3, 32)
    model.evaluate(test_ds)

    predict_image(model, test_ds, 10)


