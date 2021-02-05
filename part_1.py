import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from utils import sin, square


class rbfNN:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        # self.mus = np.linspace(min(self.x), max(self.x), self.n)
        if "sigma" not in kwargs:
            self.sigma = 0.1
        if "epochs" not in kwargs:
            self.epochs = 50
        if "lr" not in kwargs:
            self.lr = 0.2

    @staticmethod
    def calculate_rfb(x, mu_i, sigma):
        return np.exp((-(x - mu_i) ** 2) / (2 * sigma ** 2))

    def generate_rbf_nodes(self, values_x, n):
        rfb = lambda x, mui: rbfNN.calculate_rfb(x, mui, self.sigma)
        rbf_nodes = np.array([rfb(x, mu) for x in values_x for mu in self.mus])
        rbf_nodes = rbf_nodes.reshape(len(values_x), n)
        return rbf_nodes

    def train_batch(self, n):
        self.n = n
        self.init_mus()
        phi = self.generate_rbf_nodes(self.x, self.n)
        self.w = np.linalg.inv(phi.T @ phi) @ (phi.T @ self.y)

    def predict(self, x, y):
        predictions = self.generate_rbf_nodes(x, self.n) @ self.w
        total_error = mean_squared_error(y, predictions)
        return {"y": predictions, "total_error": total_error}

    def train_delta(self, n, rbf_centers = "linspace"):
        self.n = n
        self.init_mus(rbf_centers)
        w = np.random.randn(n)
        for i in range(self.epochs):
            x, y = shuffle(self.x, self.y)
            for (x_k, y_k) in list(zip(x, y)):
                phi_k = self.generate_rbf_nodes([x_k], self.n)
                w += self.lr * (y_k - phi_k @ w) @ phi_k
        self.w = w

    def compute_rbf_centers(self, x_k):
        np.linalg.norm(x_k)

    def init_mus(self, type = "linspace"):
        if type == "linspace":
            self.mus = np.linspace(min(self.x), max(self.x), self.n)
        if type == "competitive":
            pass


def batch_learning(train_data, val_data):
    residual_errors = {"train": [], "val": []}
    rbfnn = rbfNN(x = train_data['x'], y = train_data['y'])

    for n in range(1, 20):
        rbfnn.train_batch(n)
        predictions_train = rbfnn.predict(train_data['x'], train_data['y'])
        predictions_val = rbfnn.predict(val_data['x'], val_data['y'])
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


def delta_learning(train_data, val_data, lr = 0.2):
    rbfnn = rbfNN(x = train_data['x'], y = train_data['y'], lr = 0.1, epochs = 20, sigma = 0.08)
    residual_errors = {"train": [], "val": []}
    for n in range(19, 20):
        rbfnn.train_delta(n = n)
        predictions_train = rbfnn.predict(train_data['x'], train_data['y'])
        predictions_val = rbfnn.predict(val_data['x'], val_data['y'])
        residual_errors["train"].append(predictions_train["total_error"])
        residual_errors["val"].append(predictions_train["total_error"])

        plt.plot(train_data["x"], train_data["y"], label = "Train")
        plt.plot(train_data["x"], predictions_train["y"], label = "Train Predictions")
        plt.plot(val_data["x"], predictions_val["y"], label = "Val Predictions")
        plt.title(f"Delta : N. of RBF Nodes: {n}")
        plt.legend()
        plt.show()


# TODO: Import the perceptron learning from previous lab

def main():
    x_train_start, x_val_start, x_train_end, x_val_end = 0, 0.05, math.pi, math.pi

    noise = True
    train_sin = sin(x_train_start, x_train_end, 0.1, noise = noise)
    val_sin = sin(x_val_start, x_val_end, 0.1, noise = noise)
    train_square = square(x_train_start, x_train_end, 0.1, noise = noise)
    val_square = square(x_val_start, x_val_end, 0.1, noise = noise)
    # batch_learning(train_sin, val_sin)
    delta_learning(train_sin, val_sin)
    # plt.plot(train_square["x"], train_square["y"], label = "Train")
    # plt.show()


if __name__ == '__main__':
    main()
