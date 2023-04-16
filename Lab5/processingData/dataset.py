import tensorflow as tf
import pathlib
import numpy as np


class Dataset:
    def __init__(self, data_path, batch_size, val_percent, test_percent) -> None:
        self.data_path = data_path
        self.val_percent = val_percent
        self.test_percent = test_percent
        self.batch_size = batch_size

    def load_data(self, path, val_percent, test_percent):
        data_dir = pathlib.Path(path)
        data_ds = tf.data.Dataset.list_files(str(data_dir / '*/*'), shuffle=False)
        image_count = len(data_ds)
        val_size = int(image_count * val_percent)
        test_size = int(image_count * test_percent)

        train_ds = data_ds.skip(test_size + val_size)
        val_ds = data_ds.skip(test_size).take(val_size)
        test_ds = data_ds.take(test_size)

        return (train_ds, val_ds, test_ds)

    def create_train_pipeline(self, ds, preprocessor):
        image_count = len(ds)
        ds = ds.map(preprocessor.process_path, num_parallel_calls=tf.data.AUTOTUNE)\
            .batch(batch_size=self.batch_size)
        return ds

    def create_test_pipeline(self, ds, preprocessor):
        ds = ds.map(preprocessor.process_path, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size=self.batch_size)
        return ds

    def create_data_pipelines(self, preprocessor):
        (train_ds, val_ds, test_ds) = self.load_data(self.data_path, self.val_percent, self.test_percent)
        train_ds = self.create_train_pipeline(train_ds, preprocessor)
        val_ds = self.create_test_pipeline(val_ds, preprocessor)
        test_ds = self.create_test_pipeline(test_ds, preprocessor)
        return (train_ds, val_ds, test_ds)


