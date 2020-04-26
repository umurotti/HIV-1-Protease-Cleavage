#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 23:09:29 2020

@author: umurotti
"""
import os
import csv
import numpy as np

def result_bayes(root_path, trainfile_name, test_x, noOfMers, noOfAcids, laplace_alpha = 0, train_end_index = -1):
    csv_path = os.path.join(root_path, trainfile_name)
    train_x, train_y = _load_dataset(csv_path)
    #if train_end_index is not default value, set train_x to desired first n rows
    train_x = (train_x[0:train_end_index,:], train_x)[train_end_index == -1]
    train_y = (train_y[0:train_end_index], train_y)[train_end_index == -1]
    return _bayes(train_x, train_y, test_x, noOfMers, noOfAcids, laplace_alpha, train_end_index)

#bayes
def _bayes(train_x, train_y, test_x, noOfMers, noOfAcids, laplace_alpha = 0, train_end_index = -1):
    ##CALCULATE PROBABILITIES##
    #Find number of ones for each feature
    cleavable = [ row for index,row in enumerate(train_x) if train_y[index] == 1 ]
    #featureNumbers = np.matmul(unit, cleavable)
    featureNumbers = np.array(cleavable).sum(axis=0) + laplace_alpha
    #Calculate each probability of feature matrix for cleavables
    xiGivenCleavable = np.true_divide(featureNumbers, len(cleavable) + 2 * laplace_alpha)
    not_xiGivenCleavable = np.subtract(1, xiGivenCleavable)
    #Not Cleavables
    notCleavable = [ row for index,row in enumerate(train_x) if train_y[index] == 0 ]
    featureNumbers = np.array(notCleavable).sum(axis=0) + laplace_alpha
    #Calculate each probability of feature matrix for nonCleavables
    xiGivenNot_Cleavable = np.true_divide(featureNumbers, len(notCleavable) + 2 * laplace_alpha)
    not_xiGivenNot_Cleavable = np.subtract(1, xiGivenNot_Cleavable)
    #Take log to overcome overflow
    xiGivenCleavable = np.log(xiGivenCleavable)
    not_xiGivenCleavable = np.log(not_xiGivenCleavable)
    xiGivenNot_Cleavable = np.log(xiGivenNot_Cleavable)
    not_xiGivenNot_Cleavable = np.log(not_xiGivenNot_Cleavable)
    ##RUN ON TEST DATA##
    #General test data
    #Inverse value of each test value for each row
    ones = np.ones((1, train_x.shape[1]), dtype=int)
    inverse_test_x = (ones != test_x).astype(int)
    #Put zero for -Inf values in xiGivenCleavable
    InfIndices = np.where(xiGivenCleavable == np.NINF)
    xiGivenCleavable[InfIndices] = 0
    #Calculate summation of each probability for cleavables by multiplying each test column with cleavables
    res1 = np.matmul(xiGivenCleavable, np.matrix(test_x).T) + np.matmul(not_xiGivenCleavable, np.matrix(inverse_test_x).T)
    #Calculate summation of each probability for cleavables by multiplying each test column with cleavables
    res2 = np.matmul(xiGivenNot_Cleavable, np.matrix(test_x).T) + np.matmul(not_xiGivenNot_Cleavable, np.matrix(inverse_test_x).T)
    #Find probability of being cleavable and take log
    PCleavable = np.sum(train_y)/len(train_y)
    PCleavable = np.log(PCleavable)
    #Find probability of being notCleavable
    PNot_Cleavable = (len(train_y) - np.sum(train_y, dtype=int) )/len(train_y)
    PNot_Cleavable = np.log(PNot_Cleavable)
    #Checkpoint
    #Add logged probability of being cleavable and nonCleavable respectively
    res1 += PCleavable
    res2 += PNot_Cleavable
    #Choose the most probable one (argmax)
    res = np.maximum(res1, res2)
    #Create boolean vector to identify which tested values are predicted as true 
    #(if res1 is not equal element-wise, it is predicted to be false since res1 represents predicting cleavable)
    res = np.maximum(res1, res2)
    res = np.invert(np.equal(res, res2))
    exactNonCleavableRows = np.unique(np.where(test_x[:,InfIndices] == 1)[0])
    res[0, exactNonCleavableRows] = 0
    return res, res1, res2

#Load Function
def _load_dataset(filename):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
    dataset_x = np.array([[int(row[col_i]) for col_i in range(len(row)-1)] for row in dataset]) # List comprehension is used.
    dataset_y = [int(row[-1]) for row in dataset]
    return dataset_x, dataset_y