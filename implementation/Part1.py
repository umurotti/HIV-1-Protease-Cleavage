#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import Bayes as bs

def part1(root = './Dataset', trainfile = 'q2_train_set.txt', testfile = 'q2_test_set.txt'):
    #Load Datasets
    noOfMers = 8
    noOfAcids = 20
    csv_path = os.path.join(root, trainfile)
    train_x, train_y = bs._load_dataset(csv_path)
    csv_path = os.path.join(root,testfile)
    test_x, test_y = bs._load_dataset(csv_path)
    #Train
    myRes, _, _ = bs.result_bayes(root, trainfile, test_x, noOfMers, noOfAcids)
    #Find index of elements where we predicted cleavable
    trueIndices = np.where(np.array(test_y) == 1)
    #Find index of elements where we predicted nonCleavable
    falseIndices = np.where(np.array(test_y) == 0)
    #Generate results
    print("Real cleavable number:  \t", np.size(trueIndices), "\t Number predicted true cleavable:\t", np.sum(myRes[0,trueIndices]), "\t Accuracy:\t", np.sum(myRes[0,trueIndices]) / np.size(trueIndices))
    print("Real nonCleavable number:\t", np.size(falseIndices), "\t Number predicted true nonCleavable:\t", np.size(falseIndices) - np.sum(myRes[0,falseIndices]), "\t Accuracy:\t", (np.size(falseIndices) - np.sum(myRes[0,falseIndices])) / np.size(falseIndices))
    print("Total test size:\t\t", len(test_x), "\t Number predicted true in total:\t", np.sum(myRes[0,trueIndices]) + (np.size(falseIndices) - np.sum(myRes[0,falseIndices])), "\t Accuracy:\t", ((np.sum(myRes[0,trueIndices]) + (np.size(falseIndices) - np.sum(myRes[0,falseIndices])))/len(test_x)))