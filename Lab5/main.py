import tensorflow as tf

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


if __name__ == '__main__':
    dataset = Dataset(data_path=DATASET_PATH, batch_size=BATCH_SIZE, val_percent=0.15, test_percent=0.1)

    preprocessingData = PreprocessingData(CLASSES_NAMES, IMAGE_HEIGHT, IMAGE_WIDTH)
    (train_ds, validation_ds, test_ds) = dataset.create_data_pipelines(preprocessingData)
    model = train_model(train_ds, validation_ds, test_ds, 1, BATCH_SIZE)
