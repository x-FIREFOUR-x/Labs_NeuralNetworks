import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from model import InceptionV3
from processingData.preprocessingData import PreprocessingData
from processingData.dataset import Dataset

from arguments import DATASET_PATH, BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CLASSES_NAMES


def train_model(train_ds, validation_ds, test_ds, epochs, batch_size):
    model = InceptionV3((IMAGE_HEIGHT, IMAGE_WIDTH, 3), len(CLASSES_NAMES))

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

def predict_image(model, test_data, count):
    plt.figure(figsize=(4, 4))
    for (images,  correct_classes) in test_data.take(count):
        correct_classes = correct_classes.numpy()
        predictions = model.predict(images)[1]
        predicted_classes = np.argmax(predictions, axis=1)

        for index, img in enumerate(images):
            img = img.numpy()
            print('Correct =', CLASSES_NAMES[correct_classes[index]])
            print('Predicted =', CLASSES_NAMES[predicted_classes[index]])
            plt.imshow(images[index].numpy().astype('uint8'))
            plt.show()
            if (index > count):
                break


def recignize_image(model, preprocessingData, path):
    img = preprocessingData.process_single_img(path)
    predictions = model.predict(tf.expand_dims(img, axis = 0))[0]
    classes = np.argmax(predictions, axis=1).squeeze()
    print(CLASSES_NAMES[classes])
    plt.imshow(img.numpy().astype('uint8'))
    plt.show()


if __name__ == '__main__':
    dataset = Dataset(data_path=DATASET_PATH, batch_size=BATCH_SIZE, val_percent=0.15, test_percent=0.1)

    preprocessingData = PreprocessingData(CLASSES_NAMES, IMAGE_HEIGHT, IMAGE_WIDTH)
    (train_ds, validation_ds, test_ds) = dataset.create_data_pipelines(preprocessingData)

    print(tf.data.experimental.cardinality(train_ds).numpy())
    print(tf.data.experimental.cardinality(validation_ds).numpy())
    print(tf.data.experimental.cardinality(test_ds).numpy())

    model = train_model(train_ds, validation_ds, test_ds, 1, BATCH_SIZE)

    predict_image(model, test_ds, 10)

    recignize_image(model, preprocessingData, "Data\\origin\\GermanShepherd\\02.jpg")
    recignize_image(model, preprocessingData, "Data\\origin\\Others\\02.jpg")
