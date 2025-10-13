#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 16:20:16 2023

@author: ic-unicamp
"""
import argparse
import os
import fnmatch
import subprocess
import sys
#import json


def find_files(directory, file_name, baseline):
    matches = []
    for root, dirnames, filenames in os.walk(directory):
        
        if (baseline and 'baseline' in root) or (not baseline and 'baseline' not in root):
            for filename in fnmatch.filter(filenames, file_name):
                matches.append(os.path.join(root, filename))
    return matches

def main():
    
    python_command =  [sys.executable.split('/')[-1]]
    parser = argparse.ArgumentParser(description="Plotting experiment results.")
    
    parser.add_argument("--path", type=str, help="Path to the experiment results.")
    parser.add_argument("--dataset", type=str, help="dataset")
    #parser.add_argument("--folder_id", type=str, help="identifyer for folder creation", default="../new_benchmark_oml_original/meta-test/iid/20/")
    
    parser.add_argument("--folder_id", type=str, help="identifyer for folder creation", default="../new_benchmark_final_timestep_v2/meta-test/iid/20/")
    parser.add_argument("--setup", type=str, help="meta-test_baseline, meta-test_ML or meta-test_scratch", default = 'meta-test_ML')

    parser.add_argument('--scratch', help= 'run metatest from scratch', action="store_true")

    parser.add_argument('--classes_schedule', type=int, help='Number of classes to schedule.', default=2)       
    parser.add_argument('--reset_weights', action="store_true")
    parser.add_argument("--iid", action="store_true")
    parser.add_argument("--runs", type=int, default=20)
   
    args = parser.parse_args()
    args.reset_weights = True
    args.iid = True
   
    setups = [ 'meta-test_ML','meta-test_scratch','meta-test_baseline']
   
    #setups = ['meta-test_scratch','meta-test_baseline']
    
    #source_file = 'metatest_HAR_new_protocol.py'
  
    #source_file = 'metatest_HAR.py'
    
    source_file = 'metatest_HAR_benchmark_adaptive.py'
    
    source_file_plot = 'plot_metatest.py'
    
    #main_path ='/home/ic-unicamp/Meta2/abordagens/results/encoders/'    
    
    #main_path ='/home/ic-unicamp/Meta2/abordagens/new_protocol/encoders/'
    
    #main_path ='/home/ic-unicamp/Meta2/abordagens/new_benchmark/encoders/'
    
    #main_path ='/home/ic-unicamp/Meta2/abordagens/new_benchmark_oml_original/encoders/'
    main_path ='/home/ic-unicamp/Meta2/abordagens/new_benchmark_final_timestep_v2/encoders/'
    
    for scenario in setups:
    
        print('setup: ', scenario)
       
        model = []   
        encoder_path = main_path
        if scenario == 'meta-test_ML':
           baseline = False
           encoder_path = main_path     
           print('meta-learning')
           
        else:
            baseline = True
            if scenario == 'meta-test_baseline':    
                model = 'baseline'
                args.scratch = False
            else:     
                model = 'scratch'
                args.scratch = True
           
         
        file_name = 'metadata.json'
       
        result_path = find_files( encoder_path, file_name, baseline)
        
        print('result_path: ', result_path)
                
        
        for path in result_path:
            
#            if 'dsads' in path and '/nc/' in path and '10000' not in path:
#            if 'hapt' in path:
           
                if scenario == 'meta-test_ML' :
                   
                   if 'oml' in path:
                       model = 'oml'
                   elif 'maml' in path:
                       model = 'maml'
                 
                model_path = path.split(file_name)
                
                print('model path: ', model_path)
     
                arguments_list = [source_file, "--model", model, "--path", model_path[0], "--folder_id", args.folder_id, "--plot", "--name", model , "--plot_file", source_file_plot , "--runs", str(args.runs), '--classes_schedule', str(args.classes_schedule), '--new_seed']
                if args.reset_weights:
                   arguments_list.append("--reset_weights")
                if args.iid:   
                   arguments_list.append("--iid")
                if args.scratch:   
                   arguments_list.append("--scratch")
                print('arguments_list: ', arguments_list)      
                
                try:
                    result = subprocess.run(python_command + arguments_list, check=True, capture_output=False, text=True)
                    print("Command output:", result.stdout)
                except subprocess.CalledProcessError as e:
                    print("Error occurred (no replay update):", e)
                    print("Command output (if available):", e.stdout)
                    print("Command error (if available):", e.stderr)
                    
                    sys.exit()
     
     
        
if __name__ == '__main__':
    main()
