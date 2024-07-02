#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Plotting experiment results.")
    
    parser.add_argument("--path", type=str, help="Path to the experiment results.")
    
    args = parser.parse_args()
    
    path = args.path 
    
    directory = Path(path+'graphics')    

     #verify and create graphics directory
    if not directory.exists():
        # Create the directory
        directory.mkdir(parents=True, exist_ok=True)
      
    
    directory =   path + 'graphics/'  

    # open stats
    with open(path +'metadata.json', 'r') as f:
        obj = json.load(f)
    
    data = obj.get('params') 
      
    dataset = data['dataset']
    
    schedule = data['schedule']
    
    name = data['name']

    annotation = 'classes base: ' + str(data['class_base']) + '\nclasses_new: ' + str(data['class_new'])  + '\nscenario: ' +  str(data['scenario']) + '\nreset offline: ' + str(data['reset']) + '\nreset vars: ' + str(data['reset_weights']) + '\nschedule: ' + str(data['schedule']) + '\niid: ' + str(data['iid'])
   
    columns = ['Nr_classes', 'Id', 'Accuracy', 'F1 score', 'F1 score weighted', 'Precision', 'Accuracy_std', 'F1 score_std', 'F1 score weighted_std', 'Precision_std' ]
    
    keys = ['Train average stats','Test average stats','Train average stats base','Test average stats base']
    keys_id = [ 'Train', 'Test','Train base', 'Test base']
    type_id = ['meta-test','base']
    
    df = pd.DataFrame(columns=columns)
    
    for nr_classes in schedule:
        idx = 0
        for key in keys_id:
            data = obj.get('results').get(keys[idx] + ' ' + str(nr_classes))
            instance ={'Nr_classes': nr_classes, 
                        'Id': key,
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
    
    count = 0
    for i in range(0,len(keys_id),2):  
           
        id_train = keys_id[i]
        id_test = keys_id[i+1]
        
        plot_id = type_id[count]
        train = df.loc[(df['Id'] ==id_train)]
        test = df.loc[(df['Id'] ==id_test)]
        
        fig, ax = plt.subplots()
        
        #accuracy
           
        plt.errorbar(train['Nr_classes'], train['Accuracy'], yerr=train['Accuracy_std'], label='Train', marker='o', markersize=2, linestyle='-', linewidth=0.8, color='blue', capsize=1,elinewidth=0.2)
        plt.errorbar(test['Nr_classes'], test['Accuracy'], yerr=test['Accuracy_std'], label='Test', marker='o',  markersize=2, linestyle='-', linewidth=0.8, color='green', capsize=1, elinewidth=0.2)
    
        
        # Set labels and title
        plt.xlabel('Number of Classes')
        plt.ylabel('Accuracy')
        plt.title('Meta-test OML-HAR:  ' + plot_id +' - ' + dataset + ' (' + name + ')', fontsize=10)
        
        # Set x-axis tick positions and labels
        plt.xticks(df['Nr_classes'].unique().tolist())
        
        plt.ylim(0,1.5)
        # Display legend
        plt.legend()
            
        plt.text(2,1.1,annotation,fontsize=8)
        plt.savefig(directory + "accuracy_" + plot_id + ".png")
        
        
        #f1 score
    
        plt.clf()
        fig, ax = plt.subplots()
    
           
        plt.errorbar(train['Nr_classes'], train['F1 score'], yerr=train['F1 score_std'], label='Train', marker='o', markersize=2, linestyle='-', linewidth=0.8, color='blue', capsize=1,elinewidth=0.2)
        plt.errorbar(test['Nr_classes'], test['F1 score'], yerr=test['F1 score_std'], label='Test', marker='o',  markersize=2, linestyle='-', linewidth=0.8, color='green', capsize=1, elinewidth=0.2)
    
        
        # Set labels and title
        plt.xlabel('Number of Classes')
        plt.ylabel('F1 score')
        plt.title('Meta-test OML-HAR:  ' + plot_id +' - ' + dataset + ' (' + name + ')', fontsize=10)
        
        # Set x-axis tick positions and labels
        plt.xticks(df['Nr_classes'].unique().tolist())
        
        plt.ylim(0,1.5)
        # Display legend
        plt.legend()
            
        plt.text(2,1.1,annotation,fontsize=8)
        plt.savefig(directory + "f1_score_" + plot_id + ".png")  
    
        #f1 score weighted
    
        plt.clf()
        fig, ax = plt.subplots()
    
           
        plt.errorbar(train['Nr_classes'], train['F1 score weighted'], yerr=train['F1 score weighted_std'], label='Train', marker='o', markersize=2, linestyle='-', linewidth=0.8, color='blue', capsize=1,elinewidth=0.2)
        plt.errorbar(test['Nr_classes'], test['F1 score weighted'], yerr=test['F1 score weighted_std'], label='Test', marker='o',  markersize=2, linestyle='-', linewidth=0.8, color='green', capsize=1, elinewidth=0.2)
    
        
        # Set labels and title
        plt.xlabel('Number of Classes')
        plt.ylabel('F1 weighted score ')
        plt.title('Meta-test OML-HAR:  ' + plot_id +' - ' + dataset + ' (' + name + ')', fontsize=10)
        
        # Set x-axis tick positions and labels
        plt.xticks(df['Nr_classes'].unique().tolist())
        
        plt.ylim(0,1.5)
        # Display legend
        plt.legend()
            
        plt.text(2,1.1,annotation,fontsize=8)
        plt.savefig(directory + "f1weighted_" + plot_id + ".png")  
        
        
        #precision
    
        plt.clf()
        fig, ax = plt.subplots()
    
           
        plt.errorbar(train['Nr_classes'], train['Precision'], yerr=train['Precision_std'], label='Train', marker='o', markersize=2, linestyle='-', linewidth=0.8, color='blue', capsize=1,elinewidth=0.2)
        plt.errorbar(test['Nr_classes'], test['Precision'], yerr=test['Precision_std'], label='Test', marker='o',  markersize=2, linestyle='-', linewidth=0.8, color='green', capsize=1, elinewidth=0.2)
    
        
        # Set labels and title
        plt.xlabel('Number of Classes')
        plt.ylabel('Precision')
        plt.title('Meta-test OML-HAR:  ' + plot_id +' - ' + dataset + ' (' + name + ')', fontsize=10)
        
        # Set x-axis tick positions and labels
        plt.xticks(df['Nr_classes'].unique().tolist())
        
        plt.ylim(0,1.5)
        # Display legend
        plt.legend()
            
        plt.text(2,1.1,annotation,fontsize=8)
        plt.savefig(directory + "precision_" + plot_id + ".png")  
        
        count += 1
if __name__ == '__main__':
    main()