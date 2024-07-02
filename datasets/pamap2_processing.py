#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
import os
from utils import extract_column_to_txt, generate_filtered_csv, delete_file, remove_columns
from pathlib import Path


path_root = "/home/.../" # path to dataset


# generate new csv without biased activities

column_name = "activityID"  
filter_values = ["1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "12.0", "13.0", "16.0", "17.0", "24.0"] 

# train
csv_source_file_path = path_root + "train.csv"  
csv_target_file_path = path_root+ "train_tmp.csv" 

generate_filtered_csv(csv_source_file_path, column_name, filter_values, csv_target_file_path)

# test

csv_source_file_path = path_root + "test.csv"  
csv_target_file_path = path_root+ "test_prep.csv"  

generate_filtered_csv(csv_source_file_path, column_name, filter_values, csv_target_file_path)

# generate new csv without user 109 from train set

column_name = "subjectID"  
filter_values = ["1.0", "2.0", "3.0", "4.0", "7.0", "8.0"] 

# train
csv_source_file_path = path_root + "train_tmp.csv"  
csv_target_file_path = path_root+ "train_prep.csv" 

generate_filtered_csv(csv_source_file_path, column_name, filter_values, csv_target_file_path)


# create train files

source_file = path_root + "train_prep.csv"  
new_folder = 'train'

directory = Path(path_root + new_folder)    

 #verify and create graphics directory
if not directory.exists():
    # Create the directory
    directory.mkdir(parents=True, exist_ok=True)

# activity 
    
column_name = 'activityID'
target_file_name = 'y_train.txt'

target_file =   path_root + new_folder + '/' + target_file_name 

extract_column_to_txt(source_file, column_name, target_file, column_type='int')

# subject id

column_name = 'subjectID'
target_file_name = 'subject_train.txt'

target_file =   path_root + new_folder + '/' + target_file_name 

extract_column_to_txt(source_file, column_name, target_file, column_type='int')


# extract sensor data
columns_to_remove = ['Unnamed: 0','timestamp','subjectID', 'activityID']  
target_file_name = "x.txt"  # Replace with the desired output txt file path

target_file =   path_root + new_folder + '/' + target_file_name 

#remove_columns_to_txt(source_file, columns_to_remove, target_file)

remove_columns(source_file, columns_to_remove, target_file)

# create test files

source_file = path_root + "test_prep.csv"  
new_folder = 'test'

directory = Path(path_root + new_folder)    

 #verify and create graphics directory
if not directory.exists():
    # Create the directory
    directory.mkdir(parents=True, exist_ok=True)

# activity 
    
column_name = 'activityID'
target_file_name = 'y_test.txt'

target_file =   path_root + new_folder + '/' + target_file_name 

extract_column_to_txt(source_file, column_name, target_file, column_type='int')

# subject id

column_name = 'subjectID'
target_file_name = 'subject_test.txt'

target_file =   path_root + new_folder + '/' + target_file_name 

extract_column_to_txt(source_file, column_name, target_file, column_type='int')

# extract sensor data

target_file_name = "x.txt"  # Replace with the desired output txt file path

target_file =   path_root + new_folder + '/' + target_file_name 

remove_columns(source_file, columns_to_remove, target_file)

delete_file(path_root + "train_prep.csv" )
delete_file(path_root + "train_tmp.csv" )
delete_file(path_root + "test_prep.csv" )

