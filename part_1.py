import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import sys


def sin(x_range_start: float, x_range_end: float, x_step: float, noise = False):
    x = np.arange(x_range_start, x_range_end + x_step, x_step)
    noise_values = np.random.normal(0, 0.1, len(x))
    y = np.sin(2 * x) + noise_values if noise else 0
    return {"x": x, "y": y}


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


def batch_learning(train_data, val_data):
    residual_errors = {"train": [], "val": []}
    for n in range(1, 20):
        rbf_nodes_train = generate_rbf_nodes(train_data, n)
        rbf_nodes_val = generate_rbf_nodes(val_data, n)
        w = train(train_data['y'], rbf_nodes_train)
        predictions_train = predict(w, rbf_nodes_train, train_data['y'])
        predictions_val = predict(w, rbf_nodes_val, val_data['y'])
        residual_errors["train"].append(predictions_train["total_error"])
        residual_errors["val"].append(predictions_train["total_error"])
        print(f'N: {n} - Residual Error: {predictions_train["total_error"]}')
        plt.plot(train_data["x"], train_data["y"], label = "Train")
        plt.plot(train_data["x"], predictions_train["y"], label = "Train Predictions")
        plt.plot(val_data["x"], predictions_val["y"], label = "Val Predictions")
        plt.title(f"Function : N. of RBF Nodes: {n}")
        plt.legend()
        plt.show()

    # plt.plot(residual_errors["train"])
    # print(residual_errors["val"])
    # plt.plot(residual_errors["val"])
    # plt.show()


def main():
    x_train_start, x_val_start, x_train_end, x_val_end = 0, 0.05, math.pi, math.pi

    train_sin = sin(x_train_start, x_train_end, 0.1)
    val_sin = sin(x_val_start, x_val_end, 0.1)
    train_square = square(x_train_start, x_train_end, 0.1)
    val_square = square(x_val_start, x_val_end, 0.1)
    batch_learning(train_square, val_square)


if __name__ == '__main__':
    main()
