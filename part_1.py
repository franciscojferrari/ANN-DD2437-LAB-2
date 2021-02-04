import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import sys


def sin(x_range_start: float, x_range_end: float, x_step: float):
    x = np.arange(x_range_start, x_range_end + x_step, x_step)
    return {"x": x, "y": np.sin(2 * x)}


def square(x_range_start: float, x_range_end: float, x_step: float):
    x = np.arange(x_range_start, x_range_end, x_step)
    sin = np.sin(2 * x)
    sin[sin >= 0] = 1
    sin[sin < 0] = -1
    return {"x": x, "y": sin}


def calculate_rfb(x, mu_i, sigma):
    return np.exp((-(x - mu_i) ** 2) / (2 * sigma ** 2))


def generate_rbf_nodes(values, n):
    mus = np.linspace(min(values['x']), max(values['x']), n)
    sigma = 0.1
    rfb = lambda t, mui: calculate_rfb(t, mui, sigma)
    rbf_nodes = np.array([rfb(x, mu) for x in values['x'] for mu in mus])
    rbf_nodes = rbf_nodes.reshape(len(values['x']), n)
    return rbf_nodes


def train(y, rbf_nodes):
    w = np.linalg.inv(rbf_nodes.T @ rbf_nodes) @ (rbf_nodes.T @ y)
    return w


def predict(w, rbf_nodes, y):
    predictions = rbf_nodes @ w
    total_error = mean_squared_error(y, predictions)
    return {"y": predictions, "total_error": total_error}


def main():
    x_train_start, x_val_start, x_train_end, x_val_end = 0, 0.05, math.pi, math.pi

    train_sin = sin(x_train_start, x_train_end, 0.1)
    val_sin = sin(x_val_start, x_val_end, 0.1)
    train_square = square(x_train_start, x_train_end, 0.1)
    val_square = square(x_val_start, x_val_end, 0.1)

    residual_errors = {"train": [], "val": []}
    for n in range(1, 20):
        rbf_nodes_train = generate_rbf_nodes(train_sin, n)
        rbf_nodes_val = generate_rbf_nodes(val_sin, n)
        w = train(train_sin['y'], rbf_nodes_train)
        predictions_train = predict(w, rbf_nodes_train, train_sin['y'])
        predictions_val = predict(w, rbf_nodes_val, val_sin['y'])
        residual_errors["train"].append(predictions_train["total_error"])
        residual_errors["val"].append(predictions_train["total_error"])

        # plt.plot(train_sin["x"], train_sin["y"], label = "Train")
        # plt.plot(train_sin["x"], predictions_train["y"], label = "Train Predictions")
        # plt.plot(val_sin["x"], predictions_val["y"], label = "Val Predictions")
        # plt.title(f"Sin Wave : N. of RBF Nodes: {n}")
        # plt.legend()

    # plt.plot(residual_errors["train"])
    plt.plot(residual_errors["val"])
    plt.show()


if __name__ == '__main__':
    main()
