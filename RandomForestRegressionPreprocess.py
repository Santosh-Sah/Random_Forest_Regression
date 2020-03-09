# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:25:20 2020

@author: Santosh Sah
"""
from RandomForestRegressionUtils import importRandomForestRegressionDataset, saveDataSetInPickle

def preprocess():
    
    X, y = importRandomForestRegressionDataset("Random_Forest_Regression_Position_Salaries.csv")
    saveDataSetInPickle(X, y)
    

if __name__ == "__main__":
    preprocess()
