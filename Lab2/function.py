import pandas as pd
from typing import Tuple
import numpy as np


def func(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_square = x * x
    y_square = y * y
    sum = x_square + y_square
    return sum


def generate_data(path, count, min_val=0, max_val=10):
    x = np.random.rand(count) * (max_val - min_val) + min_val
    y = np.random.rand(count) * (max_val - min_val) + min_val
    result = func(x, y)
    d = {'x': x, 'y': y, 'result': result}
    df = pd.DataFrame(data=d)
    df.to_csv(path, index=False, sep=";")


def load_data(data_path):
    data = pd.read_csv(data_path, sep=';', encoding='utf-8')
    return data


def data_split(data, train_percent, val_percent, test_percent) -> Tuple:
    train_idx = int(train_percent * len(data))
    val_idx = train_idx + int(val_percent * len(data))

    train_data = np.array(data[:train_idx])
    val_data = np.array(data[train_idx:val_idx])
    test_data = np.array(data[val_idx:])

    return train_data, val_data, test_data
