import copy
import numpy as np
import multiprocessing
import random
import time

from methods.causaltree import CausalTreeRegressor

class CausalForest:
    
    def __init__(self, max_depth=-1, min_size=2, bootstrap=False, n_estimators=100, control_name=0, feature_split=0.25):
        self.bootstrap = bootstrap
        self.trees=[CausalTreeRegressor(control_name=control_name, max_depth=max_depth, min_samples_leaf=min_size, feature_split=feature_split) for i in range(n_estimators)]
        self.n_estimators = n_estimators
        
    def runIteration(self, i):
        # Bootstrap
        if self.bootstrap:
            inds_boot = np.random.choice([j for j in range(self.n)], self.n, replace = True)
            self.trees[i].fit(self.rows[inds_boot], self.treatment[inds_boot], self.labels[inds_boot])
        else:
            self.trees[i].fit(self.rows, self.treatment, self.labels)
        return self.trees[i]
        
    def fit(self, rows, labels, treatment):
        
        N = multiprocessing.cpu_count()
        self.rows = rows
        self.labels = labels
        self.treatment = treatment
        self.n = len(labels)
        
        with multiprocessing.Pool(processes = N) as p:
            results = p.map(self.runIteration, range(self.n_estimators))
            
        for i in range(self.n_estimators):
            self.trees[i] = results[i]
    
    def predict(self, X):
        for i in range(self.n_estimators):
            if i == 0:
                predictions = self.trees[i].predict(X)
            else:
                predictions += self.trees[i].predict(X)
        return predictions/self.n_estimators