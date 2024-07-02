#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from utils.utils import get_class_labels


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
            
    class_id = [obj.get('results').get('Class info').get('Class id')]   
    class_idx = [obj.get('results').get('Class info').get('Class labels')] 
    
    #data_train = [obj.get('results').get('Train')]    
    acc = obj.get('results').get('Epochs').get('Accuracy')  
    loss = obj.get('results').get('Epochs').get('Loss')
    steps = obj.get('results').get('Epochs').get('Step') 
    df_epochs = pd.DataFrame({'steps': steps, 
                              'acc_train': acc, 
                              'loss_train': loss,
                            })
      
    # training process scores
    
    
      
    ax = plt.gca()
    df_epochs.plot(kind='line',x='steps',y='acc_train', color='blue',linestyle='-', ax=ax)
    df_epochs.plot(kind='line',x='steps',y='loss_train', color='green',linestyle='-', ax=ax)

   
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title(dataset + ' - ' + obj['name'] + ' - lr (' + str(data['lr']) +')')
    #plt.ylim([0,1.55])
    plt.savefig(directory + "acc_loss_steps.png")
    
    # scores
      
    data_train = obj.get('results').get('Train average stats all')   

    data_test = obj.get('results').get('Test average stats all')   
    
    df = pd.DataFrame([{'Accuracy' : data_train['Accuracy'],
                       'F1 macro': data_train['F1_scores average'],
                       'F1_score weighted': data_train['F1_score weighted average'],
                       'Precision': data_train['Precision average'],
                       'Type' : 'Train'},
                      {'Accuracy' : data_test['Accuracy'],
                       'F1 macro': data_test['F1_scores average'],
                       'F1_score weighted': data_test['F1_score weighted average'],
                       'Precision': data_test['Precision average'],
                       'Type' : 'Test'}])
    
    bar_colors = ['paleturquoise','teal']
    
    fig, ax = plt.subplots(layout='constrained')
    
    df_transposed = df.loc[:,'Accuracy':'Precision'].T
    
    labels= df.loc[:,'Type']
    
    df_transposed.plot.bar(rot=0,color=bar_colors)
    
    # Set the labels and title
    
    plt.ylim([0,1.55])
    plt.ylabel('Scores')
    plt.title(dataset + ' - ' + obj['name'] + ' - lr (' + str(data['lr']) +')')
    plt.legend(labels)
    
    plt.savefig(directory + "scores.png")
    
   
    # Precision per class
    
    bar_colors = ['paleturquoise','teal']
    
    labels = get_class_labels(data_train['Labels'],class_id, class_idx ) 
   
   
    train_precision = data_train['Precision per class']
    
    test_precision = data_test['Precision per class']
    
    bar_width = 0.35
      
    x_train = np.arange(len(labels))
    x_test = x_train + bar_width
    
    fig, ax = plt.subplots()
    
    ax.bar(x_train, train_precision, bar_width, label='Train',color=bar_colors[0])
    
    ax.bar(x_test, test_precision, bar_width, label='Test',color=bar_colors[1])
    
    
    ax.set_xticks(x_train + bar_width / 2)
    ax.set_xticklabels(labels, fontsize=8)
    
    ax.set_ylim(0,1.4)
    
    
    ax.set_ylabel('Precision')
    ax.set_xlabel('Classes')
    ax.set_title(dataset + ' - ' + obj['name'] + ' - lr (' + str(data['lr']) +')')
    
    ax.legend()
    plt.savefig(directory + "precision_classes.png")
    
        
    # F1 - weighted
    
    train_precision = data_train['F1_scores per class']
    
    test_precision = data_test['F1_scores per class']
    
    bar_width = 0.35
      
    x_train = np.arange(len(labels))
    x_test = x_train + bar_width
    
    fig, ax = plt.subplots()
    
    ax.bar(x_train, train_precision, bar_width, label='Train',color=bar_colors[0])
    
    ax.bar(x_test, test_precision, bar_width, label='Test',color=bar_colors[1])
    
    
    ax.set_xticks(x_train + bar_width / 2)
    ax.set_xticklabels(labels, fontsize=8)
    
    ax.set_ylim(0,1.4)
    
    
    ax.set_ylabel('F1 macro')
    ax.set_xlabel('Classes')
    ax.set_title(dataset + ' - ' + obj['name'] + ' - lr (' + str(data['lr']) +')')
    
    ax.legend()
    plt.savefig(directory + "f1_classes.png")  
if __name__ == '__main__':
    main()
