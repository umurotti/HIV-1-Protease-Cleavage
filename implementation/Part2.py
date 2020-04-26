#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import Bayes as bs

def part2(root = './Dataset', trainfile = 'q2_train_set.txt', gagfile = 'q2_gag_sequence.txt'):
    def create_8mers(filename):
        with open(filename, 'r') as file:
            data = list(file.read())
            _8mer = [None] * kMers
            _8mers = [None] * (len(data) - kMers + 1)
            for char_i in range(len(data) - kMers + 1):
                for i in range(kMers):
                    _8mer[i] = data[i + char_i]
                _8mers[char_i] = _8mer
                _8mer = [None] * kMers
            return _8mers, len(data) - kMers + 1
        
    def read_amino_sequence(filename):
         with open(filename, 'r') as file:
            return list(file.read())
        
    def onehot_initialization(a):
        ncols = a.max()+1
        out = np.zeros(a.shape + (ncols,), dtype=int)
        out[all_idx(a, axis=2)] = 1
        return out
    
    def all_idx(idx, axis):
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)
    
    noOfAcids = 20
    kMers = 8
    #Load Datasets
    gag_path = os.path.join(root, gagfile)
    mers, noOfMers = create_8mers(gag_path)
    aa_names_arr = ["g", "p", "a", "v", "l", "i", "m", "c", "f", "y", "w", "h", "k",
                "r", "q", "n", "e", "d", "s", "t"]
    aa_names = dict(zip(aa_names_arr, range(len(aa_names_arr))))
    mers = np.matrix([[aa_names[x] for x in mer_i] for mer_i in mers])
    mers_encoded = onehot_initialization(mers).reshape(noOfMers, noOfAcids * kMers)
    res, res1, res2 = bs.result_bayes(root, trainfile, mers_encoded, kMers, noOfAcids, 0)
    cleavableMers = np.where(res == 1 )[1]
    cleavableIndicesPrev = cleavableMers + 3
    cleavableIndicesNext = cleavableIndicesPrev + 1
    am_seq = read_amino_sequence(gag_path)
    cleaveAminoPrev = np.array(am_seq)[cleavableIndicesPrev]
    cleaveAminoNext = np.array(am_seq)[cleavableIndicesNext]
    cleavableAminoPairs = list(map(lambda x, y:(x,y), cleaveAminoPrev, cleaveAminoNext))
    cleavableIndexPairs = list(map(lambda x, y:(x,y), cleavableIndicesPrev, cleavableIndicesNext))
    maxCleavableIndex = np.where(res1 == np.max(res1[np.where(res == 1)]))[1]
    minNonCleavableIndex = np.where(res2 ==np.min(res2[np.where(res == 0)]))[1]
    maxCleavable8mer = [aa_names_arr[x] for x in np.squeeze(np.asarray(mers[maxCleavableIndex]))]
    minNonCleavable8mer = [aa_names_arr[x] for x in np.squeeze(np.asarray(mers[minNonCleavableIndex]))]
    print("cleavableAminoPairs:\n", cleavableAminoPairs, "\ncleavableIndexPairs:\n", cleavableIndexPairs)
    print("maxCleavableIndex:\n", maxCleavableIndex, "\nminCleavableIndex:\n", minNonCleavableIndex)
    print("maxCleavable8mer:\n", maxCleavable8mer, "\nminNonCleavable8mer:\n", minNonCleavable8mer)