#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import Bayes as bs
import matplotlib.pyplot as plt

def part5(root = './Dataset', trainfile = 'q2_train_set.txt', testfile = 'q2_test_set.txt'):
    noOfAcids = 20
    kMers = 8
    csv_path = os.path.join(root, trainfile)
    train_x, train_y = bs._load_dataset(csv_path)
    csv_path = os.path.join(root, testfile)
    test_x, test_y = bs._load_dataset(csv_path)
    ##CALCULATE PROBABILITIES##
    N = len(train_x)
    #Find number of ones for each feature
    cleavable = [ row for index,row in enumerate(train_x) if train_y[index] == 1 ]
    N11 = np.array(cleavable).sum(axis=0)
    N01 = len(cleavable) - N11 
    #Not Cleavables
    notCleavable = [ row for index,row in enumerate(train_x) if train_y[index] == 0 ]
    N10 = np.array(notCleavable).sum(axis=0)
    N00 = len(notCleavable) - N10
    N1dot = N10 + N11
    N0dot = N00 + N01
    Ndot1 = len(cleavable)
    Ndot0 = len(notCleavable)
    ##
    sum_term1 = N11 * ( np.log2( ( N * N11 ) / ( N1dot * Ndot1 ) ) )
    sum_term2 = N01 * ( np.log2( ( N * N01 ) / ( N0dot * Ndot1 ) ) )
    sum_term3 = N10 * ( np.log2( ( N * N10 ) / ( N1dot * Ndot0 ) ) )
    sum_term4 = N00 * ( np.log2( ( N * N00 ) / ( N0dot * Ndot0 ) ) )
    sum_const = 1 / N
    I_UC = np.multiply( sum_const, ( sum_term1 + sum_term2 + sum_term3 + sum_term4 ) )
    ##
    I_UC[np.where(np.isnan(I_UC))] = np.Inf
    I_UC_sort_indices = np.argsort(I_UC)[::-1]
    I_UC_sorted = I_UC[I_UC_sort_indices]
    ##
    trueIndices = np.where(np.array(test_y) == 1)
    falseIndices = np.where(np.array(test_y) == 0)
    learningParams = np.empty( shape=(train_x.shape[0], 0) )
    testParams = np.empty( shape=(test_x.shape[0], 0) )
    accuracies = []
    for i in range(1, noOfAcids * kMers):
        learningParams = np.hstack((learningParams, train_x[:,I_UC_sort_indices[i-1:i]]))
        testParams = np.hstack((testParams, test_x[:,I_UC_sort_indices[i-1:i]]))
        res, _, _ = bs._bayes(learningParams, train_y, testParams, kMers, noOfAcids)
        accuracies.append(((np.sum(res[0,trueIndices]) + (np.size(falseIndices) - np.sum(res[0,falseIndices])))/len(test_x)))
    
    print("Max accuracy:\n", np.array(accuracies)[np.where(accuracies == np.max(accuracies))[0]])
    print("k = ", np.where(accuracies == np.max(accuracies))[0])
    #plot
    plt.close('all')
    plt.plot(range(1, noOfAcids * kMers), accuracies,'-k', linewidth=1)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('k')
    plt.grid(True)
    plt.title("k vs. Accuracy")
    print("Please close figures to continue...")
    plt.show()