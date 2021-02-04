import numpy as np
import pandas as pd
from typing import Tuple


def load_animals() -> Tuple[np.array, np.array]:
    data = np.genfromtxt('data_lab2/animals.dat',
                         dtype=None,
                         delimiter=',')
    data = np.resize(data,  (32, 84))
    animal_names = np.array(pd.read_csv("data_lab2/animalnames.txt", header=None)[0])
    return data, animal_names


def update_index(index: int, max_index: int) -> int:
    if index < 0:
        return max_index + index + 1
    elif index > max_index:
        return index - max_index - 1
    else:
        return index