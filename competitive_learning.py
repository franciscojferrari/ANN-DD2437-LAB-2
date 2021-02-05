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

class CompetitiveLearning(object):
    def __init__(self, num_clusters, init_type="samples", num_winners = 1):
        self.num_clusters = num_clusters
        self.init_type = init_type
        self.num_winners = num_winners
    
    def fit(self, samples, num_epochs, l_rate):
        if self.init_type == "samples":
            self.centers = self.choose_random_samples(samples)
        else:
            self.centers = self.uniform_init(samples)
        for epoch in range(num_epochs):
            self.compute_epoch(samples, l_rate)
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

"""
SMALL EXAMPLE
Let's try to find the center of a cluster that comes from a gaussian distrib'

num_samples = 100
input_dim = 2
samples = np.random.normal(loc=0.5, scale=3, size=(num_samples, input_dim))
cl = CompetitiveLearning(num_clusters=1, init_type="uniform")
centers = cl.fit(samples, num_epochs=100, l_rate=0.01)
print(centers)

plt.figure()
plt.scatter(x=samples[:, 0], y=samples[:, 1], c="black")
plt.scatter(x=centers[:, 0], y=centers[:, 1], c="yellow")

"""




