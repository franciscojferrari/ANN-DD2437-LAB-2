import numpy as np
import math
import matplotlib.pyplot as plt


def sin(x_range_start: float, x_range_end: float, x_step: float) -> np.array():
    x = np.arange(x_range_start, x_range_end + x_step, x_step)
    return {"x": x, "y": np.sin(2 * x)}


def square(x_range_start: float, x_range_end: float, x_step: float) -> np.array():
    x = np.arange(x_range_start, x_range_end + x_step, x_step)
    sin = np.sin(2 * x)
    sin[sin >= 0] = 1
    sin[sin < 0] = -1
    return {"x": x, "y": sin}


def calculate_rfb


def generate_rbf_nodes(mus: list, var: float) -> np.array():
    -()


def main():
    x_start, x_end = 0, math.pi
    train_sin = sin(x_start, x_end, `                           0.1)
    val_sin = sin(x_start, x_end, 0.05)
    train_square = square(x_start, x_end, 0.1)
    val_square = square(x_start, x_end, 0.05)

    plt.plot(train_sin["x"], train_sin["y"], train_square["x"], train_square["y"])
    plt.show()


if __name__ == '__main__':
    main()
