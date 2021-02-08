import numpy as np
import pandas as pd
import math
from typing import Tuple, List, Dict
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


def load_mp_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt("data_lab2/cities.dat", dtype=None, delimiter=",")
    data = np.resize(data, (349, 31))

    mp_district = np.genfromtxt("data_lab2/mpdistrict.dat", dtype=None, delimiter=",")
    mp_party = np.genfromtxt("data_lab2/mpparty.dat", dtype=None, delimiter=",")
    mp_sex = np.genfromtxt("data_lab2/mpsex.dat", dtype=None, delimiter=",")

    mp_sex = np.array(["blue" if i == 0 else "red" for i in mp_sex])

    # party_color_dict = {
    #     0: "magenta",
    #     1: "cyan",
    #     2: "black",
    #     3: "red",
    #     4: "brown",
    #     5: "lawngreen",
    #     6: "darkblue",
    #     7: "darkgreen",
    # }
    party_color_dict = {
        0: "blue",  # no party  - R
        1: "blue",  # m - moderate party - R
        2: "green",  # fp  - liberals - CS
        3: "red",   # s - social  democratic party - G
        4: "green",  # v - left party - CS
        5: "red",  # mp- green party - G
        6: "blue",  # kd - christian democrats  - R
        7: "green",  # c - centre party - CS
    }
    mp_party = np.array([party_color_dict[party] for party in mp_party])

    return data, mp_district, mp_party, mp_sex


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


def plot_2d_grid(title):
    x, y = [], []
    for i in range(10):
        for j in range(10):
            x.append(i)
            y.append(j)
    plt.scatter(x, y, s=450, facecolors="none", edgecolors="r")
    plt.xlim(min(x) - 1, max(x) + 1)
    plt.ylim(min(y) - 1, max(y) + 1)
    plt.title(title)
    plt.axis("off")


def plot_winner_nodes(winner_nodes: List, color_codes) -> None:
    x, y = [], []
    for winner_node in winner_nodes:
        x.append(winner_node[0])
        y.append(winner_node[1])
    x += np.random.normal(loc=0.0, scale=0.20, size=len(x))
    y += np.random.normal(loc=0.0, scale=0.20, size=len(y))
    plt.scatter(x, y, s=200, c=color_codes, alpha=0.45)
    plt.xlim(min(x) - 1, max(x) + 1)
    plt.ylim(min(y) - 1, max(y) + 1)
    plt.axis("off")
    plt.show()


def plot_occupancy(winner_nodes: List) -> None:
    x, y = [], []
    for winner_node in winner_nodes:
        x.append(winner_node[0])
        y.append(winner_node[1])
    plt.scatter(x, y, s=400, c="blue", alpha=0.01)
    plt.xlim(min(x) - 1, max(x) + 1)
    plt.ylim(min(y) - 1, max(y) + 1)
    plt.axis("off")
    plt.show()


def get_neighbor_coords(x, y) -> List:
    to_add = [[0, 1], [-1, 0], [0, -1], [1, 0]]

    return [(x + xx, y + yy) for xx, yy in to_add]


def check_coord(coords, max_index) -> bool:
    return (0 <= coords[0] <= max_index) and (0 <= coords[1] <= max_index)


def plot_animals(animal_dict: Dict) -> None:
    radius = 10
    for k, v in animal_dict.items():
        x = radius * np.cos(v/100 * 2 * math.pi)
        y = radius * np.sin(v/100 * 2 * math.pi)
        plt.annotate(k, (x, y))
        plt.scatter(x, y, s=20)
    plt.axis("off")
    plt.show()


def plot_k_best_points(k_best_points: List, color_codes) -> None:
    x, y = [], []
    colors = []
    coords = [(x, y) for x in range(10) for y in range(10)]
    for best_points, coord in zip(k_best_points, coords):
        for point in best_points:
            x.append(coord[0])
            y.append(coord[1])
            colors.append(color_codes[point])

    x += np.random.normal(loc=0.0, scale=0.15, size=len(x))
    y += np.random.normal(loc=0.0, scale=0.15, size=len(y))
    plt.scatter(x, y, s=200, c=colors, alpha=0.45)
    plt.xlim(min(x) - 1, max(x) + 1)
    plt.ylim(min(y) - 1, max(y) + 1)
    plt.axis("off")
    plt.show()
