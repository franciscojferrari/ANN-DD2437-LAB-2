import numpy as np
import matplotlib.pyplot as plt


def sin(x_range_start: float, x_range_end: float, x_step: float, noise = False):
    x = np.arange(x_range_start, x_range_end + x_step, x_step)
    noise_values = np.random.normal(0, 0.1, len(x))
    y = np.sin(2 * x) + (noise_values if noise else 0)
    return {"x": x, "y": y}


def square(x_range_start: float, x_range_end: float, x_step: float, noise = False):
    x = np.arange(x_range_start, x_range_end, x_step)
    noise_values = np.random.normal(0, 0.1, len(x))
    sin = np.sin(2 * x)
    sin[sin >= 0] = 1
    sin[sin < 0] = -1
    sin = sin + (noise_values if noise else 0)
    return {"x": x, "y": sin}
