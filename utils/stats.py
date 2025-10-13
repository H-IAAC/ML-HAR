#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 17:03:29 2023

@author: ic-unicamp
"""

import numpy as np

class Stats:
    def __init__(self, confusion_matrix=None, step=None):
             
        if confusion_matrix is None:
            self.step = step
            self.labels = []
            self.accuracy = []
            self.f1_scores = []
            self.precisions = []
            self.recalls = []
            self.support =[]
            self.f1_scores_weighted_classes = []     
            self.f1_scores_weighted = []
            self.macro_accuracy = [] 
            self.macro_recall = []
            self.macro_precision = []
            self.macro_f1 = []
            self.micro_f1 =  []
            self.precision_micro = []
            self.recall_micro = []
            
        else:    
            
            total_samples = confusion_matrix.sum()
            
            correct_predictions = np.trace(confusion_matrix)
            
            self.accuracy = correct_predictions / total_samples.sum()
            
            self.step = step
                      
            self.labels = confusion_matrix.columns
            
           
            
            num_classes = confusion_matrix.shape[0]
            self.precisions = np.zeros(num_classes)
            self.recalls = np.zeros(num_classes)
            self.f1_scores  =  np.zeros(num_classes)
    
    
            if (confusion_matrix.sum(axis=0).any()):
                for c in range(confusion_matrix.shape[0]):
                    if (confusion_matrix.iloc[:,c].sum() > 0):
                        self.precisions[c] = confusion_matrix.iloc[c,c]/confusion_matrix.iloc[:,c].sum()
            else:
                self.precisions = np.diag(confusion_matrix) / confusion_matrix.sum(axis=0)
            
                
            if (confusion_matrix.sum(axis=1).any()):
                for l in range(confusion_matrix.shape[1]):
                    if (confusion_matrix.iloc[l,:].sum() > 0):
                       self.recalls[l] = confusion_matrix.iloc[l,l]/confusion_matrix.iloc[l,:].sum()
            else:
                self.recalls = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
            
            div = self.precisions + self.recalls
            if ~div.all():
                for i in range(len(div)):
                    if (div[i]!=0):
                        self.f1_scores[i] =  2 * (self.precisions[i] * self.recalls[i]) / div[i]
            else:
               self.f1_scores = 2 * (self.precisions * self.recalls) / (self.precisions + self.recalls)
            
            
            # Support
            
            self.support = confusion_matrix.sum(axis=1)/total_samples.sum()
            
            # Weighted F1 scores
 
            self.f1_scores_weighted_classes = self.f1_scores * self.support
            self.f1_scores_weighted = np.sum(self.f1_scores * self.support)             
            
            # Macro averages
                        
            self.macro_accuracy = np.mean(self.accuracy) 
            self.macro_recall = np.mean(self.recalls) 
            self.macro_precision = np.mean(self.precisions)
            self.macro_f1 = np.mean(self.f1_scores)

            # Micro avegares       
       
            # Micro F1 (aggregated precision and recall)
            
            true_positives_sum = np.diag(confusion_matrix).sum()
                     
            false_positives_sum = np.sum(confusion_matrix.sum(axis=0) - np.diag(confusion_matrix))

            false_negatives_sum = np.sum(confusion_matrix.sum(axis=1) - np.diag(confusion_matrix))


            self.precision_micro = true_positives_sum / (true_positives_sum + false_positives_sum)
            self.recall_micro = true_positives_sum / (true_positives_sum + false_negatives_sum)
            if (self.precision_micro + self.recall_micro) > 0:
                self.micro_f1 = 2 * (self.precision_micro * self.recall_micro) / (self.precision_micro + self.recall_micro)
            else:
                self.micro_f1 = 0
            
  

    def get_f1(self, weighted=False, macro=True):
        if weighted:
           return self.f1_scores_weighted
        else:
           if macro:
              return self.macro_f1
           else:
              return self.micro_f1          
  
