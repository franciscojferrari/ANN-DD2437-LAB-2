import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from utils import sin, square
import seaborn as sns
import pandas as pd
from classification_two_layers import NueralNet


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

    def train_delta(self, n, rbf_centers = "linspace", x_val = None, y_val = None):
        self.n = n
        self.init_mus(rbf_centers)
        # self.compute_rbf_centers()
        train_losses, val_losses = [], []

        self.w = np.random.randn(n, 1)
        for i in range(self.epochs):
            x, y = shuffle(self.x, self.y)
            for (x_k, y_k) in list(zip(x, y)):
                phi_k = self.generate_rbf_nodes(x_k, self.n)
                self.w += self.lr * (y_k - phi_k @ self.w) * phi_k.T

            train_losses.append(self.predict(self.x, self.y)['total_error'])
            if type(x_val) and type(y_val) is np.ndarray:
                val_results = self.predict(x_val, y_val)
                val_losses.append(val_results['total_error'])

        return {"train_errors": train_losses, "val_errors": val_losses}

    def compute_rbf_centers(self, lr = 0.01):
        for i in range(self.epochs):
            x = shuffle(self.x)
            for x_k in x:
                distance = lambda w_i: np.linalg.norm(x_k - w_i)
                distances = np.apply_along_axis(distance, axis = 1, arr = self.mus)
                winner = np.argmin(distances)
                self.mus[winner] += lr * (x_k - self.mus[winner])

    def init_mus(self, type = "linspace"):
        if type == "linspace":
            self.mus = np.linspace(min(self.x), max(self.x), self.n)
        if type == "competitive":
            min_values = np.min(self.x, axis = 0)
            max_values = np.max(self.x, axis = 0)
            input_dim = self.x.shape[1]
            self.mus = np.random.uniform(low = min_values, high = max_values, size = (self.n, input_dim))
        if type == "samples":
            self.mus = self.x[np.random.choice(self.x.shape[0], self.n)]


# def plot_functions(*args):


def batch_learning(train_data, val_data, name = "", sigma = 0.1):
    train_error = []
    val_error = []
    rbfnn = rbfNN(x = train_data['x'], y = train_data['y'], sigma = sigma)

    for n in range(2, 35):
        rbfnn.train_batch(n)
        predictions_train = rbfnn.predict(train_data['x'], train_data['y'])
        predictions_val = rbfnn.predict(val_data['x'], val_data['y'])
        train_error.append(predictions_train["total_error"])
        val_error.append(predictions_val["total_error"])
        print(f'Batch N: {n} - Residual Error: {predictions_val["total_error"]}')
        if name == "square":
            predictions_val["y"][predictions_val["y"] > 0] = 1
            predictions_val["y"][predictions_val["y"] <= 0] = -1
            predictions_train["y"][predictions_train["y"] > 0] = 1
            predictions_train["y"][predictions_train["y"] <= 0] = -1

        plt.plot(train_data["x"], train_data["y"], label = "Train")
        plt.plot(train_data["x"], predictions_train["y"], label = "Train Predictions")
        plt.plot(val_data["x"], predictions_val["y"], label = "Val Predictions")
        plt.title(f"Batch : N. of RBF Nodes: {n}")
        plt.legend()
        # plt.show()
        plt.savefig(f"images/{name}-{n}.png")
        plt.close()
    # print(residual_errors["val"])
    plt.title("Residual Errors")
    plt.plot(train_error, label = "Train Error")
    plt.plot(val_error, label = "Val Error")
    plt.legend()
    # plt.show()
    plt.savefig(f"images/{name}-Error.png")
    plt.close()


def delta_learning(train_data, val_data, lr = 0.1, sigma = 0.1, name = ""):
    rbfnn = rbfNN(x = train_data['x'], y = train_data['y'], lr = lr, epochs = 100, sigma = sigma)
    residual_errors = {"train": [], "val": []}

    for n in range(2, 35):
        rbfnn.train_delta(n = n)
        predictions_train = rbfnn.predict(train_data['x'], train_data['y'])
        predictions_val = rbfnn.predict(val_data['x'], val_data['y'])
        residual_errors["train"].append(predictions_train["total_error"])
        residual_errors["val"].append(predictions_val["total_error"])
        print(f'Delta N: {n} - Residual Error: {predictions_val["total_error"]}')

        plt.plot(train_data["x"], train_data["y"], label = "Train")
        plt.plot(train_data["x"], predictions_train["y"], label = "Train Predictions")
        plt.plot(val_data["x"], predictions_val["y"], label = "Val Predictions")
        plt.title(f"Delta : N. of RBF Nodes: {n}")
        plt.legend()
        plt.savefig(f"images/{name}-{n}.png")
        plt.close()
        # plt.show()

    plt.title("Residual Errors")
    plt.plot(residual_errors["train"], label = "Train Error")
    plt.plot(residual_errors["val"], label = "Val Error")
    plt.legend()
    plt.savefig(f"images/{name}-Error.png")
    plt.close()
    # plt.show()


def grid_search(train_data, val_data):
    train_errors_batch, val_errors_batch = [], []
    train_errors_delta, val_errors_delta = [], []
    results = {}
    Ns = list(range(5, 37, 2))
    Sigmas = np.around(np.arange(0.1, 1.1, 0.1), 2)
    for sigma in Sigmas:
        train_error_batch, val_error_batch = [], []
        train_error_delta, val_error_delta = [], []

        rbfnn = rbfNN(x = train_data['x'], y = train_data['y'], sigma = sigma)
        for n in Ns:
            rbfnn.train_batch(n)
            predictions_train = rbfnn.predict(train_data['x'], train_data['y'])
            predictions_val = rbfnn.predict(val_data['x'], val_data['y'])
            train_error_batch.append(predictions_train["total_error"])
            val_error_batch.append(predictions_val["total_error"])

            rbfnn.train_delta(n = n)
            predictions_train = rbfnn.predict(train_data['x'], train_data['y'])
            predictions_val = rbfnn.predict(val_data['x'], val_data['y'])
            train_error_delta.append(predictions_train["total_error"])
            val_error_delta.append(predictions_val["total_error"])

        train_errors_batch.append(train_error_batch)
        val_errors_batch.append(val_error_batch)
        train_errors_delta.append(train_error_delta)
        val_errors_delta.append(val_error_delta)

    val_errors_batch = np.log(np.array(val_errors_batch))
    val_errors_delta = np.log(np.array(val_errors_delta))

    best_sigma_i, best_n_i = np.unravel_index(val_errors_batch.argmin(), val_errors_batch.shape)
    results["batch"] = {"sigma": Sigmas[best_sigma_i], "n": Ns[best_n_i]}

    best_sigma_i, best_n_i = np.unravel_index(val_errors_delta.argmin(), val_errors_delta.shape)
    results["delta"] = {"sigma": Sigmas[best_sigma_i], "n": Ns[best_n_i]}

    df_batch = pd.DataFrame(val_errors_batch.T, index = Ns, columns = Sigmas)
    df_delta = pd.DataFrame(val_errors_delta.T, index = Ns, columns = Sigmas)
    vmax_batch = np.median(np.sort(val_errors_batch.flatten())[-10:])
    vmax_batch = vmax_batch if vmax_batch < 0 else 0
    vmax_delta = np.median(np.sort(val_errors_delta.flatten())[-10:])
    vmax_delta = vmax_delta if vmax_delta < 0 else 0
    sns.heatmap(df_batch, cmap = "YlGnBu", vmax = vmax_batch)
    plt.show()
    # plt.savefig("images/gridsearch_square_batch_no_noise.png")
    # plt.close()
    sns.heatmap(df_delta, cmap = "YlGnBu", vmax = vmax_delta)
    plt.show()
    # plt.savefig("images/gridsearch_square_delta_no_noise.png")
    # plt.close()
    print(results)
    return results


def perceptron_learning(train_data, val_data):
    x_train, y_train = train_data['x'], train_data['y']
    x_val, y_val = val_data['x'], val_data['y']

    nn = NueralNet(x = x_train.T, y = y_train.T, hidden_layer_size = 29, output_layer_size = 1,
                   is_binary = False, lr = 0.0025, momentum = 0.9)
    losses = nn.train_network(8000, x_val.T, y_val.T)
    plt.plot(losses["val_losses"], label = "Validation loss")
    plt.plot(losses["epoch_losses"], label = "Train loss")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error loss")
    plt.legend()
    plt.title(f"Perceptron Learning: SQUARE MSE Train - Validate")
    plt.show()

    rbfnn = rbfNN(x = x_train, y = y_train, epochs = 100, sigma = 0.2)
    losses = rbfnn.train_delta(n = 29, x_val = x_val, y_val = y_val)
    plt.plot(losses["val_errors"], label = "Validation loss")
    plt.plot(losses["train_errors"], label = "Train loss")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error loss")
    plt.legend()
    plt.title(f"RBF Learning: SQUARE MSE Train - Validate")
    plt.show()

    predictions_val_rbf = rbfnn.predict(x = x_val, y = y_val)
    predictions_val_perceptron = nn.predict(x_val.T, y_val.T)
    plt.plot(x_val, y_val, label = "Validation Dataset")
    plt.plot(x_val, predictions_val_rbf['y'], label = "RBF Predictions")
    plt.plot(x_val, predictions_val_perceptron['pred'].T, label = "Perceptron Predictions")
    plt.title(f"Validation Dataset Predictions")
    plt.legend(loc = 'lower right')
    plt.show()


def main():
    x_train_start, x_val_start, x_train_end, x_val_end = 0, 0.05, math.pi * 2, math.pi * 2

    noise = True
    # sigma = 0.1
    train_sin = sin(x_train_start, x_train_end, 0.1, noise = noise)
    val_sin = sin(x_val_start, x_val_end, 0.1, noise = noise)
    train_square = square(x_train_start, x_train_end, 0.1, noise = noise)
    val_square = square(x_val_start, x_val_end, 0.1, noise = noise)
    # batch_learning(train_square, val_square, name = "Batch Square w. Noise")
    # delta_learning(train_square, val_square, name = "Delta Square w. Noise")

    # grid_search(train_square, val_square)
    perceptron_learning(train_square, val_square)


if __name__ == '__main__':
    main()
