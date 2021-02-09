#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 15:52:44 2021

@author: aleix
"""

import numpy as np
import matplotlib.pyplot as plt
from competitive_learning import CompetitiveLearning
from general_rbf import RBFNetwork
from sklearn.metrics import mean_squared_error

# We start with the sin2 part without noise

def get_training_test():
    start = 0
    shift = 0.05
    end = 2*np.pi
    step = 0.1
    
    num_train = int(np.round((end-start)/step + 1))
    x_train = np.linspace(start, end, num=num_train)
    x_train = np.reshape(x_train, newshape=(num_train, 1))
    y_train = np.sin(2*x_train)
    
    x_test = np.linspace(start+shift, end+shift, num_train)
    x_test = np.reshape(x_test, newshape=(num_train, 1))
    y_test = np.sin(2*x_test)
    return (x_train, y_train), (x_test, y_test)

def get_rbf_centers(x_train, num_rbfs):
    cl_epochs = 600
    cl_rate = 0.01
    cl = CompetitiveLearning(num_rbfs, init_type="samples")
    centers = cl.fit(x_train, cl_epochs, cl_rate)
    return centers

def plot_centers(centers):
    plt.figure()
    plt.title(str(centers.shape[0]) + " centers")
    for i in range(centers.shape[0]):
        plt.plot(centers[i], 0, "ro", markersize=3)
    return

num_rbfs = 11
(x_train, y_train), (x_test, y_test) = get_training_test()
centers = get_rbf_centers(x_train, num_rbfs)
# so that our rbfs span the entire space

widths = (np.pi / num_rbfs) * np.ones(num_rbfs) 
plot_centers(centers)

rbf_epochs = 100
rbf = RBFNetwork(centers, widths, output_dim=1)
train_mse = rbf.fit((x_train, y_train), (x_test, y_test), rbf_epochs, verbose=False)[0]
y_pred = rbf.predict(x_test)

plt.figure()
plt.xlabel("epoch")
plt.title("Train MSE evolution")
plt.plot(train_mse)

plt.figure()
plt.title("RBF prediction sin(2x)")
plt.plot(y_test, label="test signal")
plt.plot(y_pred, label="prediction")
plt.legend()

test_mse = mean_squared_error(y_test, y_pred)
print("test mse", test_mse)


# Now we add noise to the training and test signals
num_rbfs = 28
variance = 0.1**2
y_train += np.random.normal(scale=np.sqrt(variance), size=(y_train.shape))
y_test += np.random.normal(scale=np.sqrt(variance), size=(y_test.shape))
centers = get_rbf_centers(x_train, num_rbfs) 
plot_centers(centers)
widths = (np.pi / num_rbfs) * np.ones(num_rbfs) 

rbf_epochs = 200
rbf = RBFNetwork(centers, widths, output_dim=1)
train_mse = rbf.fit((x_train, y_train), (x_test, y_test), rbf_epochs, verbose=False)[0]
y_pred = rbf.predict(x_test)

plt.figure()
plt.xlabel("epoch")
plt.title("Train MSE evol with noise")
plt.plot(train_mse)

plt.figure()
plt.title("Noisy prediction sin(2x)")
plt.plot(y_test, label="test signal")
plt.plot(y_pred, label="prediction")
plt.legend()

test_mse = mean_squared_error(y_test, y_pred)
print("noisy test mse", test_mse)





