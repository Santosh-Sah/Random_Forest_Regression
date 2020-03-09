# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:26:53 2020

@author: Santosh Sah
"""

import pandas as pd
import pickle

"""
Import dataset and read specific column. Split the dataset in training and testing set.
Data set is very small and hence we are not going to divide the dataset in training and test set.
We will train our model on the whole dataset
"""
def importRandomForestRegressionDataset(randomForestRegressionDatasetFileName):
    
    randomForestRegressionDataset = pd.read_csv(randomForestRegressionDatasetFileName)
    X = randomForestRegressionDataset.iloc[:, 1:2].values
    y = randomForestRegressionDataset.iloc[:, 2].values
    
    """
    #spliting the dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    return X_train, X_test, y_train, y_test
    
    """
    
    return X, y

"""
Save dataset as pickle file
"""
def saveDataSetInPickle(X, y):
    
    #Write X in a picke file
    with open("X.pkl",'wb') as X_Pickle:
        pickle.dump(X, X_Pickle, protocol = 2)
    
    #Write X_test in a picke file
    with open("y.pkl",'wb') as y_Pickle:
        pickle.dump(y, y_Pickle, protocol = 2)

"""
Save RandomForestRegressionModel as a pickle file.
"""
def saveRandomForestRegressionModel(randomForestRegressionModel):
    
    #Write RandomForestRegressionModel as a picke file
    with open("RandomForestRegressionModel.pkl",'wb') as RandomForestRegressionModel_Pickle:
        pickle.dump(randomForestRegressionModel, RandomForestRegressionModel_Pickle, protocol = 2)


"""
read RandomForestRegressionModel from pickle file
"""
def readRandomForestRegressionModel():
    
    #load RandomForestRegressionModel model
    with open("RandomForestRegressionModel.pkl","rb") as RandomForestRegressionModel:
        randomForestRegressionModel = pickle.load(RandomForestRegressionModel)
    
    return randomForestRegressionModel

"""
read X from pickle file
"""
def readIndepentDataset():
    
    #load y_test
    with open("X.pkl","rb") as X_pickle:
        X = pickle.load(X_pickle)
    
    return X

"""
read y from pickle file
"""
def readDependentDataset():
    
    #load y
    with open("y.pkl","rb") as y_pickle:
        y = pickle.load(y_pickle)
    
    return y
