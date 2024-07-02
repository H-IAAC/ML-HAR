#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
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
    
    if "classes_trajectory" in data:   
     
        if data['scenario'] == 'report':
            annotation = 'classes: ' + str(data['classes']) + '\nrnd: ' +  str(data['random']) + '\nreset: ' + str(data['reset']) 
        else:
            annotation = 'trj/spt: ' + str(data['classes_trajectory']) + '\nrnd/qry: ' + str(data['classes_random'])  + '\nrnd: ' +  str(data['random']) + '\nreset: ' + str(data['reset'])
    else:
        annotation = 'traj: [481,...,962]  - qry: [0...480] ' + '\nreset: ' +  str(data['reset']) 
     
        
    # labels 
    
    class_id = [obj.get('results').get('Class info').get('Class id')]   
    class_idx = [obj.get('results').get('Class info').get('Class labels')] 
     
    # learning stats    
    
    data = [obj.get('results').get('Learning stats')]    
    dic = [item for sublist in data for item in sublist]
    df = pd.DataFrame({'step': [item['step'] for item in dic], 'acc': [item['acc'][-1] for item in dic], 'loss': [item['loss'][-1] for item in dic]})
    
    ax = plt.gca()
    df.plot(kind='line',x='step',y='loss',ax=ax)
    df.plot(kind='line',x='step',y='acc', color='red', ax=ax)
    
    maxY = max(df['loss'])
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('OML - ' + dataset + ' - ' + obj['name'])
    plt.ylim([0, maxY+1.4])
    plt.text(0.2,maxY+0.5,annotation, fontsize=8)
    plt.savefig(directory + "acc_loss.png")
    
    
    
    # train - test     
    data_train = obj.get('results').get('Train')
    data_test =  obj.get('results').get('Test')  
    data_train_complete = obj.get('results').get('Train Complete')
    data_test_complete =  obj.get('results').get('Test Complete')  
    
    df = pd.DataFrame([{'Accuracy' : data_train['Accuracy'],
                       'F1-score  macro': data_train['F1-score  macro'],
                       'F1-score micro': data_train['F1-score micro'],
                       'F1-score weighted': data_train['F1-score weighted'],
                       'Type' : 'Train'},
                      {'Accuracy' : data_test['Accuracy'],
                       'F1-score  macro': data_test['F1-score  macro'],
                       'F1-score micro': data_test['F1-score micro'],
                       'F1-score weighted': data_test['F1-score weighted'],
                        'Type' : 'Test'},
                      {'Accuracy' : data_train_complete['Accuracy'],
                       'F1-score  macro': data_train_complete['F1-score  macro'],
                       'F1-score micro': data_train_complete['F1-score micro'],
                       'F1-score weighted': data_train_complete['F1-score weighted'],
                       'Type' : 'Train Complete'},
                       {'Accuracy' : data_test_complete['Accuracy'],
                        'F1-score  macro': data_test_complete['F1-score  macro'],
                        'F1-score micro': data_test_complete['F1-score micro'],
                        'F1-score weighted': data_test_complete['F1-score weighted'],
                        'Type' : 'Test Complete'}])
    
    
    bar_colors = ['paleturquoise','teal','yellow','darkgoldenrod']
    
    
    fig, ax = plt.subplots(layout='constrained')
    
    df_transposed = df.loc[:,'Accuracy':'F1-score weighted'].T
    
    labels= df.loc[:,'Type']
    
    df_transposed.plot.bar(rot=0,color=bar_colors)
    
    # Set the labels and title
    
    plt.ylim([0, 1.4])
    plt.ylabel('Scores')
    plt.title('OML - ' + dataset + ' - ' + obj['name'])

    plt.legend(labels,fontsize="8" )
    plt.text(0.7,1.1,annotation)
    
    plt.savefig(directory + "scores.png")
    
    # F1-scores per class
    train_f1_scores = []
    labels_training = obj.get('params').get('label_training')
    labels_training.sort(key=int)

    for i in labels_training:
        idx = data_train['Labels'].index(i)
        train_f1_scores.append(data_train['F1-scores per class'][idx])
    
    test_f1_scores = []    
    for i in labels_training:
        idx = data_test['Labels'].index(i)
        test_f1_scores.append(data_test['F1-scores per class'][idx])    
    
    
    labels = get_class_labels(labels_training,class_id, class_idx )   
    
   
    bar_width = 0.35
    
    x_train = np.arange(len(labels))
    x_test = x_train + bar_width
    
    fig, ax = plt.subplots()
    
    
    ax.bar(x_train, train_f1_scores, bar_width, label='Train',color=bar_colors[0])
    
    ax.bar(x_test, test_f1_scores, bar_width, label='Test',color=bar_colors[1])
    
    ax.set_xticks(x_train + bar_width / 2)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.4)
    ax.legend()
    ax.set_ylabel('F1-score')
    ax.set_xlabel('Classes')
    ax.set_title('OML - ' + dataset + ' - ' + obj['name'])
    
    
    plt.text(0.7,1.1,annotation)
    plt.savefig(directory + "f1_scores.png")
    
    #  F1-scores per class complete dataset
    
      
    labels = get_class_labels(data_train_complete['Labels'],class_id, class_idx )   
    
    train_f1_scores = data_train_complete['F1-scores per class']
    
    test_f1_scores = data_test_complete['F1-scores per class']
    
    bar_width = 0.35
    
    x_train = np.arange(len(labels))
    x_test = x_train + bar_width
    
    fig, ax = plt.subplots()
    
    ax.bar(x_train, train_f1_scores, bar_width, label='Train Complete',color=bar_colors[0])
    
    ax.bar(x_test, test_f1_scores, bar_width, label='Test Complete',color=bar_colors[1])
    
    
    ax.set_xticks(x_train + bar_width / 2)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1.4)
    ax.legend()
    ax.set_ylabel('F1-score')
    ax.set_xlabel('Classes')
    ax.set_title('OML - ' + dataset + ' - ' + obj['name'])
    
    plt.text(0.7,1.1,annotation)
    
    plt.savefig(directory + "f1_scores_complete.png")
    
    # Precision per class
    
    bar_colors = ['paleturquoise','teal']
    
    
    # F1-scores per class
    train_precision = []
    labels_training = obj.get('params').get('label_training')
    labels_training.sort(key=int)

    for i in labels_training:
        idx = data_train['Labels'].index(i)
        train_precision.append(data_train['Precision per class'][idx])
    
    test_precision = []    
    for i in labels_training:
        idx = data_test['Labels'].index(i)
        test_precision.append(data_test['Precision per class'][idx])    
    
    
    labels = get_class_labels(labels_training,class_id, class_idx )   
    
      
    bar_width = 0.35
    
    
    x_train = np.arange(len(labels))
    x_test = x_train + bar_width
    
    fig, ax = plt.subplots()
    
    ax.bar(x_train, train_precision, bar_width, label='Train',color=bar_colors[0])
    
    ax.bar(x_test, test_precision, bar_width, label='Test',color=bar_colors[1])
    
    
    ax.set_xticks(x_train + bar_width / 2)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.4)
    
    
    ax.set_ylabel('Precision')
    ax.set_xlabel('Classes')
    ax.set_title('OML - ' + dataset + ' - ' + obj['name'])
    
    ax.legend()
    plt.text(0.7,1.1,annotation)
    plt.savefig(directory + "precision.png")
    
    
    # Precision per class complete
    
    bar_colors = ['paleturquoise','teal']
          
    labels = get_class_labels(data_train_complete['Labels'],class_id, class_idx )   

    
    train_precision = data_train_complete['Precision per class']
    
    test_precision = data_test_complete['Precision per class']
    
    bar_width = 0.35
    
    
    x_train = np.arange(len(labels))
    x_test = x_train + bar_width
    
    fig, ax = plt.subplots()
    
    ax.bar(x_train, train_precision, bar_width, label='Train Complete',color=bar_colors[0])
    
    ax.bar(x_test, test_precision, bar_width, label='Test Complete',color=bar_colors[1])
    
    
    ax.set_xticks(x_train + bar_width / 2)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1.4)
    
    
    ax.set_ylabel('Precision')
    ax.set_xlabel('Classes')
    ax.set_title('OML - ' + dataset + ' - ' + obj['name'])
    
    ax.legend()
    plt.text(0.7,1.1,annotation)
    plt.savefig(directory + "precision_complete.png")

if __name__ == '__main__':
    main()