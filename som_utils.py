import numpy as np
import pandas as pd
from typing import Tuple, Any
import matplotlib.pyplot as plt


def load_animals() -> Tuple[np.array, np.array]:
    data = np.genfromtxt("data_lab2/animals.dat", dtype=None, delimiter=",")
    data = np.resize(data, (32, 84))
    animal_names = np.array(pd.read_csv("data_lab2/animalnames.txt", header=None)[0])
    return data, animal_names


def load_cities() -> np.array:
    data = np.genfromtxt("data_lab2/cities.dat", dtype=None, delimiter=",")
    data = np.resize(data, (10, 2))
    return data


def load_mp_data() -> np.array:
    data = np.genfromtxt("data_lab2/cities.dat", dtype=None, delimiter=",")
    data = np.resize(data, (349, 31))
    return data


def update_index(index: int, max_index: int) -> int:
    if index < 0:
        return max_index + index + 1
    elif index > max_index:
        return index - max_index - 1
    else:
        return index


def plot_city_data(data):
    """Plot the locations of the various cities."""
    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x, y, s=100, marker="X")
    plt.xlim(min(x) - 0.2, max(x) + 0.2)
    plt.ylim(min(y) - 0.2, max(y) + 0.2)


def plot_weight_data(data):
    """Plot the weights from the SOM Network and the links between the nodes."""
    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x, y)
    plt.plot(x, y, "red")
    plt.plot([x[0], x[-1]], [y[0], y[-1]], "red")
    plt.xlim(min(x) - 0.2, max(x) + 0.2)
    plt.ylim(min(y) - 0.2, max(y) + 0.2)
    plt.show()


def get_neighbor_coords(x, y):
    to_add = [[0, 1], [-1, 0], [0, -1], [1, 0]]

    return [(x + xx, y + yy) for xx, yy in to_add]


def check_coord(coords, max_index):
    return (0 <= coords[0] <= max_index) and (0 <= coords[1] <= max_index)
