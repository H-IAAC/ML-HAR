#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import pandas as pd
import os
import fnmatch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

 
def find_files(directory, file_name, dataset):
    matches = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, file_name):
             matches.append(os.path.join(root, filename))
    return matches


def plot_data(data, metric, dataset, path_id, aug):
    
    
    index = pd.MultiIndex.from_tuples([
    ('Test', 'baseline'), ('Test', 'maml'), ('Test', 'oml'), ('Test', 'scratch'),
    ('Train', 'baseline'), ('Train', 'maml'), ('Train', 'oml'), ('Train', 'scratch')
     ], names=['key', 'Name'])

    df = pd.DataFrame(data, index=index)

    # Extracting the data for plotting
    test_data = df.loc['Test']
    train_data = df.loc['Train']
    test_data = test_data.rename(index={'baseline': 'batch'})
    train_data = train_data.rename(index={'baseline': 'batch'})
    test_data = test_data.rename(index={'oml': 'OML'})
    train_data = train_data.rename(index={'oml': 'OML'})
    test_data = test_data.rename(index={'maml': 'MAML-Rep'})
    train_data = train_data.rename(index={'maml': 'MAML-Rep'})   

    # Setting up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_colors = ['paleturquoise','teal']

    # Creating bars with error bars (standard deviation)
    bar_width = 0.35
    index = np.arange(len(test_data))
    opacity = 0.7
    elinewidth = 1  # Set the error bar line width
    capsize = 2     # Set the width of the error bar caps

    bars_train = ax.bar(index - bar_width/2, train_data[metric], bar_width,
                        yerr=train_data[metric+' std'], alpha=opacity, label='Train', color=bar_colors[0],error_kw={'elinewidth': elinewidth, 'capsize': capsize})
    bars_test = ax.bar(index + bar_width/2, test_data[metric], bar_width,
                       yerr=test_data[metric+' std'], alpha=opacity, label='Test', color=bar_colors[1],error_kw={'elinewidth': elinewidth, 'capsize': capsize})




    # Adding labels, title, and customizing ticks
    #ax.set_xlabel('Model',fontsize=14)
    ax.set_ylabel(metric,fontsize=18)
    ax.set_title(dataset.upper(),fontsize=20)
    ax.set_xticks(index)
    ax.set_ylim(0,1.2)
    ax.set_xticklabels(test_data.index,fontsize=16)
    ax.legend(fontsize=18)

    plt.savefig(path_id + "/" + dataset +'_'+ aug  +'_'+  metric + '.svg', bbox_inches='tight')
    

def main():

 
    ref_datasets = ['ucihar']
    
    # main path of results to plot
    main_path ='/home/.../' 
    
    file_name = 'metadata.json'
     
    # directory to save plots 
    ref = 'paper/'
    
    
    for dataset in ref_datasets:
        
  
        path = main_path + dataset
    
        result_path = find_files(path, file_name, dataset)  
        
        columns = ['Name', 'Nr_classes', 'F1 micro', 'F1 macro', 'Precision', 'F1 micro std', 'F1 macro std', 'Precision std' ]
        
        keys = ['Train average stats','Test average stats']
 
        keys_id = [ 'Train', 'Test']

        
        df = pd.DataFrame(columns=columns)

        for path in result_path:
        
            with open(path, 'r') as f:
                obj = json.load(f)
    
            data = obj.get('params') 
           
            schedule = data['schedule']
            
            name = data['name']
            aug = data.get('augmentation_ref')
            for nr_classes in schedule:
                idx = 0
                for key in keys_id:
                    name = obj.get('name')
                    data = obj.get('results').get(keys[idx] + ' ' + str(nr_classes))
                    
                    
                    instance ={ 'Name' : name,
                                'Nr_classes': nr_classes, 
                                'key': key,
                                'Aug': aug,
                                'F1 micro': data['Accuracy'],
                                'F1 macro': data['F1-score  macro'],
                                'Precision': data['Macro precision'],
                                'F1 micro std': data['Accuracy std'], 
                                'F1 macro std': data['F1-score  macro std'], 
                                'Precision std': data['Macro precision std']}
                                           
                    new_df = pd.DataFrame([instance])
                    df = pd.concat([df,new_df], ignore_index=True) 
                    idx += 1
            
                      
        
        for aug in np.unique(df['Aug']):
            
            directory = Path(main_path + ref + dataset + '/' + aug)    

            if not directory.exists():
               directory.mkdir(parents=True, exist_ok=True)
                         
            df_plot = df.loc[(df['Aug']==aug)]
            
                        
 
            # Plotting accuracy
            data_plot_accuracy = df_plot.groupby(['key','Name'])[['F1 micro', 'F1 micro std']].mean()
            plot_data(data_plot_accuracy, 'F1 micro', dataset, str(directory),aug)
            
            # Plotting F1 score
            data_plot_f1 = df_plot.groupby(['key','Name'])[['F1 macro', 'F1 macro std']].mean()
            plot_data(data_plot_f1,'F1 macro', dataset, str(directory), aug)
            
            # Plotting Precision
            data_plot_precision = df_plot.groupby(['key','Name'])[['Precision', 'Precision std']].mean()
            plot_data(data_plot_precision,'Precision' , dataset, str(directory), aug)
            
 
if __name__ == '__main__':
    main()