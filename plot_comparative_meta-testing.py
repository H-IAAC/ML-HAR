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


def plot_data(data_plot, ylabel, ax, i, dsc):
    
    colors = ['blueviolet' ,'orange',   'lime', 'black', 'blue','lightgray', 'blueviolet' ]
    
    data_plot = data_plot.reset_index()
    grouped = data_plot.groupby('Name')
    color_id = 0
    for name, group in grouped:
        if name == 'baseline':
           name = 'batch'
        elif name== 'oml':
             name = name.upper()
        else:
            if name == 'maml':
               name = 'MAML-Rep'  

        ax.errorbar(group['Nr_classes'], group[ylabel], yerr=group[ylabel+'_std'], label=name, markersize=1, linestyle='-', color=colors[color_id], linewidth=0.8,  elinewidth=0.3)
        color_id += 1
    ax.set_xlabel('Number of Classes',fontsize=12)
    ax.set_ylabel('Scores', fontsize=12)
    if ylabel == 'Accuracy':
       title = 'F1 micro'
    elif ylabel == 'F1 score':
       title = 'F1 macro'
    else:   
       title = 'Precision'
    if i in [1,2,4,5]:
        ax.yaxis.set_visible(False) 
  
    ax.set_ylim(0, 1.1)  # Adjust ylim as needed
    plt.text(0.95, 0.95,dsc, ha='right', va='top', transform=ax.transAxes,fontsize=12)

    ax.set_xticks(data_plot['Nr_classes'].unique().tolist())
    
    if i in range(0,3):
       ax.set_title(title)
       ax.xaxis.set_visible(False) 
    

def main():

    
    ref_datasets = ['dsads','ucihar', 'pamap2', 'hapt']
    
    # path to results to plot      
    main_path ='/home/.../' 
    
    file_name = 'metadata.json'
     
    # directory to save plots 
    ref = 'comparative/'
    
    
    for dataset in ref_datasets:
        
  
        path = main_path + dataset
    
        result_path = find_files(path, file_name, dataset)  
        
        columns = ['Name', 'Nr_classes', 'Id', 'Accuracy', 'F1 score', 'F1 score weighted', 'Precision', 'Accuracy_std', 'F1 score_std', 'F1 score weighted_std', 'Precision_std' ]
        # all
        keys = ['Train average stats','Test average stats','Train average stats base','Test average stats base']
        keys_id = [ 'Train', 'Test','TrainBase', 'TestBase']
        
       
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
                                'Id': key,
                                'Aug': aug,
                                'Accuracy': data['Accuracy'],
                                'F1 score': data['F1-score  macro'],
                                'F1 score weighted': data['F1-score weighted'],
                                'Precision': data['Macro precision'],
                                'Accuracy_std': data['Accuracy std'], 
                                'F1 score_std': data['F1-score  macro std'], 
                                'F1 score weighted_std': data['F1-score weighted std'], 
                                'Precision_std': data['Macro precision std']}
                                           
                    new_df = pd.DataFrame([instance])
                    df = pd.concat([df,new_df], ignore_index=True) 
                    idx += 1
            
                      
        
        for aug in np.unique(df['Aug']):
            
            directory = Path(main_path + ref + dataset + '/' + aug)    

            if not directory.exists():
               directory.mkdir(parents=True, exist_ok=True)
            fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))  
                        

            fig.subplots_adjust(wspace=0.02, hspace=0.02)

            row = 0
            for i in range(0,len(keys_id)):  
                   
                id_plot = keys_id[i]
               
                
                df_plot = df.loc[(df['Id'] ==id_plot) & (df['Aug']==aug)]
                
                # Plotting accuracy
                data_plot_accuracy = df_plot.groupby(['Name', 'Nr_classes'])[['Accuracy', 'Accuracy_std']].mean()
                data_plot_accuracy = data_plot_accuracy.sort_values(by='Name')
                plot_data(data_plot_accuracy, 'Accuracy', axs[i][0], i+row, id_plot)
                
                # Plotting F1 score
                data_plot_f1 = df_plot.groupby(['Name', 'Nr_classes'])[['F1 score', 'F1 score_std']].mean()
                data_plot_f1 = data_plot_f1.sort_values(by='Name')
                plot_data(data_plot_f1, 'F1 score', axs[i][1], i+1+row, id_plot)
                
                # Plotting Precision
                data_plot_precision = df_plot.groupby(['Name', 'Nr_classes'])[['Precision', 'Precision_std']].mean()
                data_plot_precision = data_plot_precision.sort_values(by='Name')
                plot_data(data_plot_precision, 'Precision', axs[i][2],i+2+row, id_plot)
                
                row = 2
            # Add legend without border
            # Combine legends
            
            handles, labels = axs[0, 0].get_legend_handles_labels()
    
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.001), frameon=False, fancybox=True, shadow=True, ncol=len(labels),fontsize = 14)
            plt.suptitle(dataset.upper(),x=0.5, y=0.94, fontsize = 14)
            plt.savefig(str(directory) + "/" + dataset+'_'+aug +'600', dpi=600, bbox_inches='tight')
            plt.savefig(str(directory) + "/" + dataset+'_'+aug +'.pdf', bbox_inches='tight')
            plt.savefig(str(directory) + "/" + dataset+'_'+aug +'.svg', bbox_inches='tight')
            

if __name__ == '__main__':
    main()
