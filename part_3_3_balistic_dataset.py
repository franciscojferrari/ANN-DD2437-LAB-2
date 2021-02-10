#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 18:27:12 2021

@author: aleix
"""

import numpy as np
import matplotlib.pyplot as plt
from competitive_learning import CompetitiveLearning
from general_rbf import RBFNetwork
from matplotlib.patches import Circle
from sklearn.metrics import mean_squared_error

# the file has the structure from balist.dat
def get_inputs_targets(filename):
    f = open(filename, "r")
    x_inputs = []
    y_targets = []
    for line in f:
        x_in, y_tar = line.split("\t")
        x1, x2 = x_in.split(" ")
        y1, y2 = y_tar.split(" ")
        x_inputs.append([float(x1), float(x2)])
        y_targets.append([float(y1), float(y2)])
    return np.array(x_inputs), np.array(y_targets)


def plot_receptive_field(x_train, centers, widths):
    num_rbfs = centers.shape[0]
    fig, axes = plt.subplots()
    axes.set_title("Receptive field")
    axes.set_xlabel("angle")
    axes.set_ylabel("velocity")
    plt.plot(x_train[:, 0], x_train[:, 1], "ko", markersize=3, label="data points")
    plt.plot([], [], "ro", label="rbf centers")
    for i in range(num_rbfs):
        circle = Circle(xy=(centers[i, 0], centers[i, 1]), radius=widths[i])
        axes.add_patch(circle)
        axes.plot(centers[i, 0], centers[i, 1], "ro", markersize=5)
    plt.legend()
    return  fig, axes      

def plot_prediction(y_test, y_pred):
    num_points = y_test.shape[0]
    plt.figure()
    plt.title("RBF prediction")
    plt.xlabel("distance")
    plt.ylabel("height")
    plt.plot(y_test[0, 0], y_test[0, 1], "go", markersize=3, label="True")
    plt.plot(y_pred[0, 0], y_pred[0, 1], "ro", markersize=3, label="Prediction")
    for i in range(1, num_points):
        plt.plot(y_test[i, 0], y_test[i, 1], "go", markersize=3)
        plt.plot(y_pred[i, 0], y_pred[i, 1], "ro", markersize=3)
    plt.legend()
    return

x_train, y_train = get_inputs_targets("data_lab2/ballist.dat")
x_test, y_test = get_inputs_targets("data_lab2/balltest.dat")
    
num_rbfs = 10
cl = CompetitiveLearning(num_rbfs, init_type="uniform")
centers = cl.fit(x_train, num_epochs=300, l_rate=0.01)
sigma = np.sqrt(1 / (num_rbfs*np.pi))
widths = sigma * np.ones(num_rbfs)
plot_receptive_field(x_train, centers, widths)



rbf = RBFNetwork(centers, widths, output_dim=2)
train_mse = rbf.fit((x_train, y_train), (x_test, y_test), num_epochs=100, verbose=False)[0]

plt.figure()
plt.plot(train_mse)

y_pred = rbf.predict(x_test)
test_mse = mean_squared_error(y_test, y_pred)
 
print("num rbfs = ", num_rbfs)
print("Test mse:", test_mse)
plot_prediction(y_test, y_pred)






