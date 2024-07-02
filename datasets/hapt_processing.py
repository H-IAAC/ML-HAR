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

from sklearn.preprocessing import Normalizer, MinMaxScaler
from utils import create_directory

SAMPLING_FREQ = 50 # Hz

SLIDING_WINDOW_LENGTH = int(2.56*SAMPLING_FREQ)

sensors = 6

SLIDING_WINDOW_STEP = int(SLIDING_WINDOW_LENGTH/2)


def standardize(mat):
    """ standardize each sensor data columnwise"""
    for i in range(mat.shape[1]):
        mean = np.mean(mat[:, [i]])
        std = np.std(mat[:, [i]])
        mat[:, [i]] -= mean
        mat[:, [i]] /= std

    return mat


def __rearrange(a,y,s, window, overlap):
    l, f = a.shape
    shape = (int( (l-overlap)/(window-overlap) ), window, f)
    stride = (a.itemsize*f*(window-overlap), a.itemsize*f, a.itemsize)
    X = np.lib.stride_tricks.as_strided(a, shape=shape, strides=stride)
    #import pdb; pdb.set_trace()

    l,f = y.shape
    shape = (int( (l-overlap)/(window-overlap) ), window, f)
    stride = (y.itemsize*f*(window-overlap), y.itemsize*f, y.itemsize)
    Y = np.lib.stride_tricks.as_strided(y, shape=shape, strides=stride)
    Y = Y.max(axis=1)

    l,f = s.shape
    shape = (int( (l-overlap)/(window-overlap) ), window, f)
    stride = (s.itemsize*f*(window-overlap), s.itemsize*f, s.itemsize)
    S = np.lib.stride_tricks.as_strided(s, shape=shape, strides=stride)
    S = S.max(axis=1)


    return X, Y.flatten(), S.flatten()

def normalize(data):
    """ l2 normalization can be used"""

    y = data[:, 0].reshape(-1, 1)
    X = np.delete(data, 0, axis=1)
    transformer = Normalizer(norm='l2', copy=True).fit(X)
    X = transformer.transform(X)

    return np.concatenate((y, X), 1)


def normalize_df(data):
    """ l2 normalization can be used"""

    #y = data[:, 0].reshape(-1, 1)
    #X = np.delete(data, 0, axis=1)
    transformer = Normalizer(norm='l2', copy=True).fit(data)
    data = transformer.transform(data)

    return data

def min_max_scaler(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    return data

def read_dir(DIR, user_test):

    folder1=sorted(os.listdir(DIR))
    #import pdb; pdb.set_trace()

    labels = np.genfromtxt(os.path.join(DIR,folder1[-1]), delimiter=' ')
    accel_files = folder1[:int(len(folder1[:-1])/2)]
    gyro_files = folder1[int(len(folder1[:-1])/2):-1]

    train_d = []
    test_d = []
    test_subject_d = []
    train_subject_d = []
    for a_file,g_file in zip(accel_files,gyro_files):
        #import pdb; pdb.set_trace()
        a_ff = os.path.join(DIR, a_file)
        g_ff = os.path.join(DIR, g_file)
        a_df = np.genfromtxt(a_ff, delimiter=' ')
        g_df = np.genfromtxt(g_ff, delimiter=' ')
        ss = a_file.split('.')[0].split('_')
        exp, user = int(ss[1][-2:]), int(ss[2][-2:])

        indices = labels[labels[:,0]==exp]
        indices = indices[indices[:,1]==user]
        for ii in range(len(indices)):
            a_sub = a_df[int(indices[ii][-2]):int(indices[ii][-1]),:]
            g_sub = g_df[int(indices[ii][-2]):int(indices[ii][-1]),:]
            subject_id = np.full(len(a_sub), user)
            if user in user_test:
                test_d.extend(np.append(np.append(a_sub,g_sub,axis=1),np.array([indices[ii][-3]]*len(a_sub))[:,None],axis=1))
                test_subject_d.extend(subject_id)
            else:
                train_d.extend(np.append(np.append(a_sub,g_sub,axis=1),np.array([indices[ii][-3]]*len(a_sub))[:,None],axis=1))
                train_subject_d.extend(subject_id)

    train_x = np.array(train_d)[:,:-1]
    test_x = np.array(test_d)[:,:-1]
    train_y = np.array(train_d)[:,-1]
    test_y = np.array(test_d)[:,-1]
    train_subject = np.array(train_subject_d)
    test_subject = np.array(test_subject_d)

    print('\nunique Y ', np.unique(train_y),np.unique(test_y))
    print('\nunique subject', np.unique(train_subject), np.unique(test_subject))

    train_x = normalize(train_x)
    test_x = normalize(test_x)
    train_x, train_y, subject_train = __rearrange(train_x, train_y.astype(int).reshape((-1,1)),train_subject.astype(int).reshape((-1,1)), SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    test_x, test_y, subject_test = __rearrange(test_x, test_y.astype(int).reshape((-1,1)),test_subject.astype(int).reshape((-1,1)), SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
   
    return train_x, train_y, test_x, test_y, subject_train, subject_test
    
if __name__ == "__main__": 

    path_root = '/home/.../' # path to datasets
    path = path_root + 'RawData'
    
    user_tst = np.arange(1,11)
    train_data, train_labels, test_data, test_labels, train_subject, test_subject = read_dir(path, user_tst)
    
    print("Train Data: {}".format(np.shape(train_data)))
    print("Test Data: {}".format(np.shape(test_data)))
    print("Train labels: {}".format(np.shape(train_labels)))
    print("Test labels: {}".format(np.shape(test_labels)))
    print("Train subject: {}".format(np.shape(train_subject)))
    print("Test labels: {}".format(np.shape(test_subject )))
    
    train_X = train_data.reshape((len(train_data), int(sensors * SLIDING_WINDOW_LENGTH)))
    test_X = test_data.reshape((len(test_data), int(sensors * SLIDING_WINDOW_LENGTH)))

    assert len(test_data) == len(test_labels)
    assert len(train_data) == len(train_labels)
    assert len(train_data) == len(train_subject)
    assert len(test_data) == len(test_subject)

    print('reshape\n')
    print("Train Data X: {}".format(np.shape(train_X)))
    print("Test Data X: {}".format(np.shape(test_X)))
    print(train_labels.astype(int))
    
    # train
    
    # Calculate the decimal precision of the floats in the array
    decimal_precision = len(str(np.max(np.abs(train_X))).split('.')[1])

    # Construct the format specifier based on the decimal precision
    format_specifier = f"%.{decimal_precision}f"
    create_directory(path_root + 'train') 
    print('root', path_root + 'train/X_train.txt' )
    np.savetxt(path_root + '/train/X_train.txt', train_X, fmt=format_specifier) 
    np.savetxt(path_root + '/train/y_train.txt', train_labels, fmt="%d")
    np.savetxt(path_root + '/train/subject_train.txt', train_subject, fmt="%d")

    
    # test
    # Calculate the decimal precision of the floats in the array
    decimal_precision = len(str(np.max(np.abs(test_X))).split('.')[1])

    # Construct the format specifier based on the decimal precision
    format_specifier = f"%.{decimal_precision}f"
    create_directory(path_root + 'test') 
    print('root', path_root + 'test/X_test.txt' )
    np.savetxt(path_root + '/test/X_test.txt', test_X, fmt=format_specifier) 
    np.savetxt(path_root + '/test/y_test.txt', test_labels, fmt="%d")
    np.savetxt(path_root + '/test/subject_test.txt', test_subject, fmt="%d")

    
