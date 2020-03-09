# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:27:31 2020

@author: Santosh Sah
"""

import matplotlib.pyplot as plt
import numpy as np
from RandomForestRegressionUtils import (readRandomForestRegressionModel, readIndepentDataset, readDependentDataset)

"""
Visualising the Random Forest Regression results (for higher resolution and smoother curve)

"""
def visualisingRandomForestRegressionInHighResolution():
    
    X = readIndepentDataset()
    y = readDependentDataset()
    randomForestRegressionModel = readRandomForestRegressionModel()
    
    X_grid = np.arange(min(X), max(X), 0.1)
    X_grid = X_grid.reshape((len(X_grid), 1))    

    # Visualising the Polynomial Regression results
    plt.scatter(X, y, color = 'red')
    plt.plot(X_grid, randomForestRegressionModel.predict(X_grid), color = 'blue')
    plt.title('Truth or Bluff (Polynomial Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    
    plt.savefig("random_forest__trainingsetresult_high_resolution.png")
    
    plt.show()
    
if __name__ == "__main__":
    visualisingRandomForestRegressionInHighResolution()