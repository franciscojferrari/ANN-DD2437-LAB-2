#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 11:39:40 2021

@author: aleix
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class RBFNetwork(object):
    def __init__(self, rbf_centers, rbf_widths, output_dim):
        self.rbf_centers = rbf_centers
        self.rbf_widths = rbf_widths
        self.output_dim = output_dim
        self.num_rbfs = rbf_centers.shape[0]
        self.weights = np.random.normal(size=(self.output_dim, self.num_rbfs))
    
    # train data is a tuple (x_train, y_train)
    # x_train has shape num_samples, sample_dim
    # y_train has shape num_samples, output_dim
    def fit(self, train_data, val_data, num_epochs, l_rate=0.01):
        x_train, y_train = train_data
        x_val, y_val = val_data
        self.x_train_rbfs = self.compute_rbfs(x_train)
        self.x_val_rbs = self.compute_rbfs(x_val)
        train_mses = []
        val_mses = []
        for epoch in range(num_epochs):
            t_mse, v_mse = self.compute_epoch(y_train, y_val, l_rate)
            train_mses.append(train_mses)
            val_mses.append(v_mse)
        return train_mses, val_mses
    
    def predict(self, x_inputs):
        x_rbfs = self.compute_rbfs(x_inputs)
        return  x_rbfs @ self.weights.T # shape (num_inputs, output_dim)
    
    def compute_epoch(self, y_train, y_val, l_rate):
        num_samples = self.x_train_rbfs.shape[0]
        for i in range(num_samples):
            x_rbf = np.copy(self.x_train_rbfs[i]) # shape num_rbfs
            err_array = y_train - self.weights @ x_rbf # shape output_dim
            # reshape in order to prepare the matrix structure for weights 
            err_array = np.reshape(err_array, newshape=(self.output_dim, 1))
            x_rbf = np.reshape(x_rbf, newshape=(1, self.num_rbfs))
            self.weights += l_rate * (err_array * x_rbf)
            # matrix of shape output_dim, num_rbfs
        y_train_pred = self.x_train_rbfs @ self.weights.T
        y_val_pred = self.x_val_rbs @ self.weights.T
        train_mse = mean_squared_error(y_train, y_train_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)
        return train_mse, val_mse
    
    # we call the increase_dimension function for each sample
    def compute_rbfs(self, x_inputs):
        num_samples = x_inputs.shape[0]
        x_rbfs = np.zeros(shape=(num_samples, self.num_rbfs))
        for i in range(num_samples):
            x_rbfs[i] = self.increase_dimension(x_inputs[i])
        return x_rbfs # shape (num_samples, num_rbfs)
    
    def increase_dimension(self, x_input):
        x_rbf = np.zeros(self.num_rbfs)
        for i in range(self.num_rbfs):
            mu = self.rbf_centers[i]
            sigma = self.rbf_widths[i]
            x_rbf[i] = rbf_function(x_input, mu, sigma)
        return x_rbf # shape num_rbfs
    
def rbf_function(x, mu, sigma):
    coef = np.sum((x-mu)**2) / (2*sigma**2)
    return np.exp(coef)
