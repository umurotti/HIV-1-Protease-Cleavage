#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import Bayes as bs
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation


    
def part6(root = './Dataset', trainfile = 'q2_train_set.txt'):
    def rotate(angle):
        ax.view_init(azim=angle)    
    csv_path = os.path.join(root, trainfile)
    train_x, train_y = bs._load_dataset(csv_path)
    no_of_rows, _ = train_x.shape
    centroid = np.mean(train_x, axis = 0)
    std = np.std(train_x, axis = 0)
    Z = (train_x - centroid) / std
    Z_transpose = Z.T
    #covariance matrix with up to a constant k
    cov_mat_wk = np.matmul(Z_transpose, Z)
    eig_values, eig_col_vectors = np.linalg.eig(cov_mat_wk)
    idx = eig_values.argsort()[::-1]   
    eig_values_sorted = eig_values[idx]
    eig_col_vectors_sorted = eig_col_vectors[:,idx]
    Z_centered = np.matmul(Z, eig_col_vectors_sorted)
    PC1 = Z_centered[:,0]
    PC2 = Z_centered[:,1]
    PC3 = Z_centered[:,2]
    PVE = np.sum(eig_values_sorted[0:3]) / np.sum(eig_values)
    print("PVE: ", PVE)
    #plot
    plt.close('all')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(PC1, PC2, PC3, c=PC1, linewidth=0.1)
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    ax.view_init(azim=50)
    animation.FuncAnimation(fig, rotate, frames=np.arange(0,365,1),interval=0.1)
    print("Please close figures to continue...")
    plt.show()