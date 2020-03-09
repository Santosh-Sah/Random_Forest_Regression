# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:24:42 2020

@author: Santosh Sah
"""

import pandas as pd
from RandomForestRegressionUtils import readRandomForestRegressionModel

def predictRandomForestRegression():
    
    randomForestRegressionModel = readRandomForestRegressionModel()
    
    inputValue = [6.5]
    inputValueDataframe = pd.DataFrame(inputValue)
    
    predictedValue = randomForestRegressionModel.predict(inputValueDataframe.values)
    
    print(predictedValue)

if __name__ == "__main__":
    predictRandomForestRegression()

