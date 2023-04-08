import pandas as pd
import numpy as np
import random
from typing import Tuple
import sys

import numpy as np
import random


def func(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    sum = x + y
    sum_square = sum * sum
    return sum_square


def generate_data(path, count, min_val=0, max_val=100):
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
