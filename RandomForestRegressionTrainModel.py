# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:26:05 2020

@author: Santosh Sah
"""
from sklearn.ensemble import RandomForestRegressor
from RandomForestRegressionUtils import (saveRandomForestRegressionModel, readIndepentDataset, readDependentDataset)

"""
Train random forest regression model 
"""
def trainRandomForestRegressionModel():
    
    X = readIndepentDataset()
    y = readDependentDataset()
    
    # Fitting RandomForest Regression to the dataset
    randomForestRegressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
    randomForestRegressor.fit(X, y)
    
    saveRandomForestRegressionModel(randomForestRegressor)

if __name__ == "__main__":
    trainRandomForestRegressionModel()
