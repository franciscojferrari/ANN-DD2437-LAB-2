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
        If random: the centers are initialized as a uniform distribution 
        that tries to cover the sample space.
        If samples: random samples are chosen as center inits.
    num_winners: int; A way to ensure that all centers will be updated.
        The first num_winners will be updated.  
"""

import numpy as np

class CompetitiveLearning(object):
    def __init__(self, num_clusters, init_type="samples", num_winners = 1):
        self.num_clusters = num_clusters
        self.init_type = init_type
        self.num_winners = num_winners
    
    def fit(self, samples, num_epochs, l_rate):
        self.input_dim = samples.shape[1]
        if self.init_type == "samples":
            self.centers = self.choose_random_samples(samples)
        else:
            self.centers = self.uniform_init(samples)
        for epoch in range(num_epochs):
            self.compute_epoch(samples, l_rate)
        return self.centers
    
    def compute_epoch(self, samples, l_rate):
        for sample in samples:
            self.centers = self.update_winners(sample, l_rate)
        return
    
    def update_winners(self, sample, l_rate):
        return
    
    def uniform_init(self, samples):
        return
    
    def choose_random_samples(samples, self):
        return

