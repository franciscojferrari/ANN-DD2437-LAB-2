#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 13:01:39 2021

@author: aleix
The Competitive Learning class implements a clustering algorithm.
It finds the centers of clusters in data via the fit method.

Parameters:
    num_clusters: int; 
    init_type: str; How the algorithm initializes the centers of the clusters.
        init_type can be: "random" or "samples". 
        If random: the centers are initialized from a uniform distribution 
        that tries to cover the sample space.
        If samples: random samples are chosen as center inits.
    num_winners: int; A way to aviod the dead unit problem.
        The first num_winners will be updated.  
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class CompetitiveLearning(object):
    def __init__(self, num_clusters, init_type="samples", num_winners = 1):
        self.num_clusters = num_clusters
        self.init_type = init_type
        self.num_winners = num_winners
    
    def fit(self, samples, num_epochs, l_rate):
        self.num_epochs = num_epochs
        if self.init_type == "samples":
            self.centers = self.choose_random_samples(samples)
        else:
            self.centers = self.uniform_init(samples)
        for epoch in range(num_epochs):
            self.compute_epoch(samples, l_rate)
            filename = "Animation/" + self.init_type + "_" + \
                                       str(self.num_winners) + "_" + str(epoch)
            np.save(file= filename, arr=self.centers)
        return self.centers
    
    def compute_epoch(self, samples, l_rate):
        for sample in samples:
            self.update_winners(sample, l_rate)
        return
    
    def update_winners(self, sample, l_rate):
        indexes_winners = find_closest_centers(sample, self.centers, 
                                                              self.num_winners)
        for idx in indexes_winners:
            self.centers[idx, :] += l_rate * (sample - self.centers[idx, :])  
        return
    
    def uniform_init(self, samples):
        num_samples, input_dim = samples.shape
        # 1d arrays: minimum and maximum values found for each input dimension
        min_values = np.min(samples, axis=0)
        max_values = np.max(samples, axis=0)
        self.centers = np.zeros(shape=(self.num_clusters, input_dim))
        for dim in range(input_dim):
            min_val = min_values[dim]
            max_val = max_values[dim]
            self.centers[:, dim] = np.random.uniform(low=min_val, high=max_val,
                                                        size=self.num_clusters)
        return self.centers
    
    def choose_random_samples(self, samples):
        num_samples = samples.shape[0]
        sample_indexes = [i for i in range(num_samples)]
        # random.sample returns k non repated indexes chosen randomly 
        random_indexes = random.sample(sample_indexes, k=self.num_clusters)
        random_samples = samples[random_indexes]
        return random_samples
    
    # Methods for the animation
    def update_frame(self, centers):
        self.centers_line.set_xdata(centers[:, 0])
        self.centers_line.set_ydata(centers[:, 1])
        self.title.set_text("Epoch " + str(self.epoch) + "/ " + str(self.num_epochs))
        return
        
    def frames(self):
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            filename = filename = "Animation/" + self.init_type + "_" + \
            str(self.num_winners) + "_" + str(epoch) + ".npy"
            centers = np.load(filename)
            yield centers
    
    def initialize_plot(self, samples):
        self.centers_line = plt.plot([], [], "yo" ,markersize=5)[0]
        self.title = self.centers_line.axes.text(x=2, y=13, s="")
        return
    
        
# closest in the sense of euclidean distance
# if num_closest = 1, the closest center is returned.
# if num_closest = 2, the 2 closest centers are returned, etc
def find_closest_centers(sample, centers, num_closest):
    indexes_closest = np.zeros(num_closest, dtype=np.int32)
    # 1d array, sample square distance respect to each center
    square_distances = np.sum((centers-sample)**2, axis=1)
    for i in range(num_closest):
        idx_closest = np.argmin(square_distances)
        indexes_closest[i] = idx_closest
        # to avoid finding the same minimum in the next iter
        square_distances[idx_closest] = np.inf
    return indexes_closest


def create_2d_clusters(num_clusters, points_per_cluster, range_limit, seed=30):
    np.random.seed(seed)
    total_points = num_clusters * points_per_cluster
    points = np.zeros(shape=(total_points, 2))
    for cluster in range(num_clusters):
        # each cluster is gaussian distributed
        mu_x = np.random.uniform(low=-range_limit, high=range_limit)
        mu_y = np.random.uniform(low=-range_limit, high=range_limit)
        sigma = 0.5 # I choose a tight Gaussian
        points[cluster*points_per_cluster: (cluster+1)*points_per_cluster, 0] = \
            np.random.normal(loc=mu_x, scale=sigma, size=(points_per_cluster))
            
        points[cluster*points_per_cluster: (cluster+1)*points_per_cluster, 1] = \
            np.random.normal(loc=mu_y, scale=sigma, size=(points_per_cluster))
    np.random.seed()
    return points

"""
SMALL EXAMPLE
Let's try to find the center of a cluster that comes from a gaussian distrib'
"""

num_clusters = 6
points_per_cluster = 30
range_limit = 10
input_dim = 2

init_type = "uniform"
num_winners = 3
l_rate = 0.01
num_epochs = 30

samples = create_2d_clusters(num_clusters,points_per_cluster, range_limit)
cl = CompetitiveLearning(num_clusters, init_type, num_winners)
centers = cl.fit(samples, num_epochs, l_rate)

fig2 = plt.figure()
plt.scatter(x=samples[:, 0], y=samples[:, 1], c="black")
plt.scatter(x=centers[:, 0], y=centers[:, 1], c="yellow")


fig = plt.figure()
plt.scatter(x=samples[:, 0], y=samples[:, 1], c="black")
anim = FuncAnimation(fig, func=cl.update_frame, frames=cl.frames, 
                     init_func=cl.initialize_plot(samples), interval=500)



