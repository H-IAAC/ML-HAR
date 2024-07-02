# !/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thur Jun 10 2021

@author: Rebecca Adaimi

HAPT dataset loading and preprocessing
Participants 29 and 30 used as test data 
"""

import numpy as np
import pandas as pd
from utils import create_directory


    # load the dataset, returns train and test X and y elements
def load_dataset(root='', image_set='train'):
    # load all train
    X = load_dataset_group(image_set, root)
   
    return X

def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = pd.read_csv(prefix + name, header=None, delim_whitespace=True)           
        loaded.append(data)
    X_data = np.concatenate(loaded, axis=1)
    
    return X_data     

    # load a dataset group, such as train or test
    # borrowed methods from the tutorial
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial-Signals/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_' + group + '.txt', 'total_acc_y_' + group + '.txt', 'total_acc_z_' + group + '.txt']
    # body acceleration
    filenames += ['body_acc_x_' + group + '.txt', 'body_acc_y_' + group + '.txt', 'body_acc_z_' + group + '.txt']
    # body gyroscope
    filenames += ['body_gyro_x_' + group + '.txt', 'body_gyro_y_' + group + '.txt', 'body_gyro_z_' + group + '.txt']
    # load input data
    X = load_group(filenames, filepath)

    return X
 
if __name__ == "__main__": 

    path_root = '/home/...datasets/ucihar/'  

    train_X = load_dataset(path_root, 'train')


    print("Train Data: {}".format(np.shape(train_X)))
    
    create_directory(path_root + 'train') 
    print('root', path_root + 'train/X_train.txt' )
    np.savetxt(path_root + '/train/X_train.txt', train_X)

    test_X = load_dataset(path_root, 'test')


    print("Train Data: {}".format(np.shape(test_X)))
    
    create_directory(path_root + 'test') 
    print('root', path_root + 'test/X_test.txt' )
    np.savetxt(path_root + '/test/X_test.txt', test_X)

