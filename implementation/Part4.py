#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import Bayes as bs
import numpy as np
import matplotlib.pyplot as plt

def calculate_accurracy(root, noOfAcids, kMers, train_file, test_file, laplace_alpha, train_end_index = -1):
    csv_path = os.path.join(root, test_file)
    test_x, test_y = bs._load_dataset(csv_path)
    res, _, _ = bs.result_bayes(root, train_file, test_x, kMers, noOfAcids, laplace_alpha, train_end_index)
    #Find index of elements where we predicted cleavable
    trueIndices = np.where(np.array(test_y) == 1)
    #Find index of elements where we predicted nonCleavable
    falseIndices = np.where(np.array(test_y) == 0)
    #Generate results
    accuracy = ((np.sum(res[0,trueIndices]) + (np.size(falseIndices) - np.sum(res[0,falseIndices])))/len(test_x))
    return accuracy

def part4(root = './Dataset', trainfile = 'q2_train_set.txt', testfile = 'q2_test_set.txt'):    
    noOfAcids = 20
    kMers = 8
    accuracies = [calculate_accurracy(root, noOfAcids, kMers, trainfile, testfile, laplace_alpha) for laplace_alpha in range(11)]
    print("Accuracy values for whole train set:\n", accuracies)
    accuracies_75 = [calculate_accurracy(root, noOfAcids, kMers, trainfile, testfile, laplace_alpha, 75) for laplace_alpha in range(11)]
    print("Accuracy values for first 75 eleemnts of train set:\n", accuracies_75)
    #plot
    plt.close('all')
    plt.plot(range(11), accuracies,'-ko', label= "Full train data")
    plt.plot(accuracies_75, '--r^', label= "First 75 rows of train data")
    plt.xticks(np.arange(min(range(11)), max(range(11))+1, 1.0))
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Alpha')
    plt.grid(True)
    plt.title("Alpha vs. Accuracy")
    plt.legend()
    plt.figure()
    plt.plot(range(11), accuracies,'-ko', label= "Full train data")
    plt.xticks(np.arange(min(range(11)), max(range(11))+1, 1.0))
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Alpha')
    plt.grid(True)
    plt.title("Alpha vs. Accuracy (Full Train Set)")
    plt.legend()
    plt.figure()
    plt.plot(accuracies_75, '--r^', label= "First 75 rows of train data")
    plt.xticks(np.arange(min(range(11)), max(range(11))+1, 1.0))
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Alpha')
    plt.grid(True)
    plt.title("Alpha vs. Accuracy (First 75)")
    plt.legend()
    print("Please close figures to continue...")
    plt.show()