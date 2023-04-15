import tensorflow as tf
from tensorflow import keras


def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (227, 227))
    return image, label


def get_data():
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

    validation_images, validation_labels = train_images[:5000], train_labels[:5000]
    train_images, train_labels = train_images[5000:], train_labels[5000:]

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

    train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()

    train_ds = (train_ds
                .map(process_images)
                .shuffle(buffer_size=train_ds_size)
                .batch(batch_size=32, drop_remainder=True))
    test_ds = (test_ds
               .map(process_images)
               .shuffle(buffer_size=train_ds_size)
               .batch(batch_size=32, drop_remainder=True))
    validation_ds = (validation_ds
                     .map(process_images)
                     .shuffle(buffer_size=train_ds_size)
                     .batch(batch_size=32, drop_remainder=True))

    return train_ds, validation_ds, test_ds
