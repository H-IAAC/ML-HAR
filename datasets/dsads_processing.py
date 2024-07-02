# !/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thur Jun 10 2021

@author: Rebecca Adaimi

HAPT dataset loading and preprocessing
Participants 29 and 30 used as test data 
"""

import numpy as np
import os
import pandas as pd

from utils import create_directory

SAMPLING_FREQ = 25 # Hz

SLIDING_WINDOW_LENGTH = int(5*SAMPLING_FREQ)

sensors = 9
sensors_units = 5

sensors_columns = int(sensors * sensors_units)


 
if __name__ == "__main__": 

    path_root = '/home/.../datasets/dsads/'  
    path = path_root + 'data'
    activities = sorted(os.listdir(path))

    train_data = []
    test_data = []
    train_labels = []
    test_labels = []
    train_subjects = []
    test_subjects = []

    for a in activities:
        activity_path = os.sep.join((path,a))
        participants = sorted(os.listdir(activity_path))

        test_participants = participants[-2:]
        train_participants = participants[:-2]
        print('train_participants ',train_participants )
        print('test_participants ',test_participants )
        for p in train_participants:
            train_data_sub = []
            full_path = os.sep.join((activity_path, p))

            segments = sorted(os.listdir(full_path))
            for seg in segments:
                segment_path = os.sep.join((full_path, seg))
                data = pd.DataFrame(np.genfromtxt(segment_path, delimiter=','))
                data = data[~np.isnan(data).any(axis=1)]   
                train_data_sub.extend(np.reshape(np.array(data),(1,np.shape(data)[0], np.shape(data)[1])))
                train_labels.extend([int(a[-2:])])
                train_subjects.extend([int(p[-1:])])
            train_data.extend(train_data_sub)

        for p in test_participants:
            test_data_sub = []
            full_path = os.sep.join((activity_path, p))
            print('p ', p)
            segments = sorted(os.listdir(full_path))
            for seg in segments:
                segment_path = os.sep.join((full_path, seg))
                #print(segment_path)
                data = pd.DataFrame(np.genfromtxt(segment_path, delimiter=','))
                data = data[~np.isnan(data).any(axis=1)]   
                test_data_sub.extend(np.reshape(np.array(data),(1,np.shape(data)[0], np.shape(data)[1])))
                test_labels.extend([int(a[-2:])])  
                test_subjects.extend([int(p[-1:])])
            test_data.extend(test_data_sub)


    print("Train Data: {}".format(np.shape(train_data)))
    print("Train Labels: {}".format(np.shape(train_labels)))
    print("Train Subjects: {}".format(np.shape(train_subjects)))
    print("Test Data: {}".format(np.shape(test_data)))
    print("Test Labels: {}".format(np.shape(test_labels)))
    print("Test subjects: {}".format(np.shape(test_subjects)))

    assert len(test_data) == len(test_labels)
    assert len(train_data) == len(train_labels)
    assert len(train_data) == len(train_subjects)
    assert len(test_data) == len(test_subjects)

    train_x = np.array(train_data)
    test_x = np.array(test_data)
    train_y = np.array(train_labels)
    test_y = np.array(test_labels)
    train_subject = np.array(train_subjects)
    test_subject = np.array(test_subjects)
    
    assert len(train_x) == len(train_y)
    assert len(test_x) == len(test_y)
    assert len(train_x) == len(train_subject)
    assert len(test_x) == len(test_subject)
    
    print("np array")
    
    print("Train Data: {}".format(np.shape(train_x)))
    print("Train Labels: {}".format(np.shape(train_y)))
    print("Train Subjects: {}".format(np.shape(train_subject)))
    print("Test Data: {}".format(np.shape(test_x)))
    print("Test Labels: {}".format(np.shape(test_y)))
    print("Test subjects: {}".format(np.shape(test_subject)))

   
    train_X = train_x.reshape((len(train_x), int(sensors_columns * SLIDING_WINDOW_LENGTH)))
    test_X = test_x.reshape((len(test_x), int(sensors_columns * SLIDING_WINDOW_LENGTH)))

    print("reshape") 
    print("Train Data: {}".format(np.shape(train_X)))
    print("Train Labels: {}".format(np.shape(test_X)))
    
    print('\nunique  Y ', np.unique(train_y),np.unique(test_y))
    print('\nunique subject', np.unique(train_subject), np.unique(test_subject))


   # train 
   # Calculate the decimal precision of the floats in the array
    decimal_precision = len(str(np.max(np.abs(train_X))).split('.')[1])
    
    print('\ndecimal precision', decimal_precision)

    # Construct the format specifier based on the decimal precision
    format_specifier = f"%.{decimal_precision}f"
    print('\format_specifier', format_specifier)
    create_directory(path_root + 'train') 
    print('root', path_root + 'train/X_train.txt' )
    #np.savetxt(path_root + '/train/X_train.txt', train_X, fmt=format_specifier) 
    np.savetxt(path_root + '/train/X_train.txt', train_X, fmt="%f") 
    np.savetxt(path_root + '/train/y_train.txt', train_y, fmt="%d")
    np.savetxt(path_root + '/train/subject_train.txt', train_subject, fmt="%d")

   
    # test
    # Calculate the decimal precision of the floats in the array
    decimal_precision = len(str(np.max(np.abs(test_X))).split('.')[1])

    # Construct the format specifier based on the decimal precision
    format_specifier = f"%.{decimal_precision}f"
    create_directory(path_root + 'test') 
    print('root', path_root + 'train/X_test.txt' )
    #np.savetxt(path_root + '/test/X_test.txt', test_X, fmt=format_specifier) 
    np.savetxt(path_root + '/test/X_test.txt', test_X, fmt="%f") 
    np.savetxt(path_root + '/test/y_test.txt', test_y, fmt="%d")
    np.savetxt(path_root + '/test/subject_test.txt', test_subject, fmt="%d")
 