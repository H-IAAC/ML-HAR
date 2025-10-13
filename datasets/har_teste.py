#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 14:49:21 2025

@author: ic-unicamp
"""

import os
import copy
import torch
import numpy as np
import datasets.datasetfactory as df
from torch.utils.data import DataLoader, Dataset
from utils.helpers import load_file
from typing import Optional
from datasets.augmentation import TimeSeriesAugmenter



DIR = os.path.abspath(os.path.dirname(__file__))

def get_dataset(dataset):
    """Return the correct dataset."""
    dataset = dataset.lower()
    try:
        DATASETS_DICT = df.DatasetFactory.get_dataset_conf(dataset)[0]
        return eval(DATASETS_DICT[dataset])
    except KeyError:
        raise ValueError("Unkown dataset: {}".format(dataset))

def get_dataset_param(dataset):
    """Return the correct dataset."""
    dataset = dataset.lower()
    try:
        return df.DatasetFactory.get_dataset_conf(dataset)[0]
    except KeyError:
        raise ValueError("Unkown dataset: {}".format(dataset))
        
def get_dataloaders(dataset, dataset_path, root=None, shuffle=True, is_train=True, pin_memory=True, num_workers=4,
                    batch_size=128, is_standardized=True, subject=all, dataloader=True,  standardize_mode="channel", time_cut_ratio: float = None, eps: float = 1e-8, data_augmentation=None,  **kwargs):
   
    pin_memory = pin_memory and torch.cuda.is_available() # only pin if GPU available
    #pin_memory = False
    Dataset = get_dataset(dataset)
    if root is None:
        dataset = Dataset(dataset = dataset,
                          is_train=is_train,
                          root = dataset_path,
                          standardize_mode=standardize_mode,
                          time_cut_ratio = time_cut_ratio,
                          eps=eps,
                          is_standardized=is_standardized)
    else:
        dataset = Dataset(dataset = dataset,
                          root=root,
                          is_train=is_train,
                          standardize_mode=standardize_mode,
                          time_cut_ratio = time_cut_ratio,
                          eps=eps,
                          is_standardized=is_standardized)

    if data_augmentation is not None:     
        
       augmenter = TimeSeriesAugmenter(time_axis=2, channel_axis=1) 

       augmenter.augment_dataset(dataset, data_augmentation, append_to_dataset=True)

    if dataloader:
        return DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=num_workers,
                          pin_memory=pin_memory,
                          **kwargs)
    else:
        return dataset

    
class HumanActivityRecognition(Dataset):
    """
    Human activity recognition dataset (IMU/HAR)
    Shape das entradas: (N, C, T) = (num_janelas, num_canais, pontos_no_tempo)
    """
    def __init__(
        self,
        dataset,
        root=None,
        is_train=True,
        is_standardized=False,
        standardize_mode: str = "channel",     # "channel" ou "window"
        time_cut_ratio: float = None,
        eps: float = 1e-8 
    ):
        config = get_dataset_param(dataset)
        self.sensores = config["sensors"]
        self.time_window = config["time_window"]
        self.freq = config["freq"]
        self.data_points = int(self.time_window * self.freq)   
        self.data_size = (self.sensores, int(self.data_points))# (C, T)
        self.labels_id = config["labels_id"]
 
        self.standardize_mode = standardize_mode
        self.is_standardized = is_standardized
        self.time_cut_ratio = time_cut_ratio,
        self.eps=eps,
        
        mu = np.float32(0.0)
        sd = np.float32(0.0)

        self.time_cut_ratio = time_cut_ratio


        if root is None:
            root = config["path"]
            self.dataset_path = config["path"]

        image_set = 'train' if is_train else 'test'

        if is_standardized:
            
            data_train = self.load_dataset(root, 'train')            
            X_train_np = data_train[0].reshape(len(data_train[0]), self.sensores, int(self.data_points))
            Y_train_np = data_train[1]
            classes_id_train = data_train[2]
            subject_id_train = data_train[3]

            if image_set == 'train':
                
                X_np, mu, sd = self.standardize_dataset(
                    X_train = X_train_np,
                    X_test=None,
                    mode=self.standardize_mode,
                    time_cut_ratio=self.time_cut_ratio,
                    eps=self.eps
                )
                Y_np = Y_train_np
                classes_id = classes_id_train
                subject_id = subject_id_train

            else:  # image_set == 'test'
                data_test = self.load_dataset(root, 'test')                
                X_test_np = data_test[0].reshape(len(data_test[0]), self.sensores, int(self.data_points))
                Y_test_np = data_test[1]
                classes_id_test = data_test[2]
                subject_id_test = data_test[3]

                # Padroniza o teste usando stats do treino (mode="channel")
                # ou por janela, de forma independente (mode="window")
                X_np, mu, sd = self.standardize_dataset(
                    X_train = X_train_np,
                    X_test=X_test_np,
                    mode=self.standardize_mode,
                    time_cut_ratio=self.time_cut_ratio,
                    eps=self.eps
                )
                
                Y_np = Y_test_np
                classes_id = classes_id_test
                subject_id = subject_id_test
        else:
            data = self.load_dataset(root, image_set)
            X_np = data[0].reshape(len(data[0]), self.sensores, int(self.data_points))
            Y_np = data[1]
            classes_id = data[2]
            subject_id = data[3]

        self.X = torch.from_numpy(np.asarray(X_np)).float()          # (N, C, T)
        self.Y = torch.from_numpy(np.asarray(Y_np)).flatten().long()

        self.classes_id = classes_id
        self.nr_classes = len(classes_id)
        self.labels = np.arange(0, len(classes_id))
        self.subject_id = [str(i) for i in np.unique(subject_id)]
        self.mu = mu
        self.sd = sd
        self.nr_samples = self.X.shape[0]
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input = self.X[idx,:,:]
        target = self.Y[idx]

        return input, target
    
    def get_data_size(self):
        """Return the correct data size."""
        return self.data_size

    def get_num_classes(self):
        "Return the number of classes"
        return self.nr_classes

    def get_subject_id(self):
        "Return the number of classes"
        return self.subject_id

    def get_class_labels(self):
        """Return the class labels"""
        return self.labels

    def get_classes_id(self):
        """Return the class original label id"""
        return self.classes_id

    def get_class_id(self,label):
        """Return the class id original labels"""
        position = np.where(self.labels == label)[0]
        
        return self.classes_id[position]

    def get_dataset_path(self):
        """Return the class id original labels"""
                
        return self.dataset_path

    def get_class_description(self,label):
        """Return the class id original labels"""
        position = np.where(self.labels == label)[0]
        key = self.classes_id[position][0]
       
        return  self.labels_id[key]

    def get_subject_id(self):
        """Return the class id original labels"""
        return self.subject_id
        
    def split_data(self, split_percentage: float):
        """
        Splits the dataset into two new, separate HumanActivityRecognition objects.

        The original dataset is shuffled before splitting to ensure random distribution.

        Args:
            split_percentage (float): The percentage of data for the first group (e.g., 0.8 for an 80/20 split).
                                      Must be a value between 0.0 and 1.0.

        Returns:
            tuple: A tuple containing two new HumanActivityRecognition objects (group1, group2).
        """
        if not (0.0 < split_percentage < 1.0):
            raise ValueError("split_percentage must be a float strictly between 0.0 and 1.0.")

        nr_samples = self.X.shape[0]
        
        # 1. Create a random permutation of indices to shuffle the data
        shuffled_indices = torch.randperm(nr_samples)

        # 2. Calculate the split point
        split_point = int(nr_samples * split_percentage)
        if split_point == 0 or split_point == nr_samples:
             raise ValueError("split_percentage results in an empty dataset. Adjust the value.")

        # 3. Get indices for each of the two new groups
        indices1 = shuffled_indices[:split_point]
        indices2 = shuffled_indices[split_point:]

        # 4. Create two new objects as deep copies of the original.
        # This preserves all metadata (config, mu, sd, etc.).
        group1 = copy.deepcopy(self)
        group2 = copy.deepcopy(self)

        # 5. Assign the corresponding data splits to the new objects
        group1.X = self.X[indices1]
        group1.Y = self.Y[indices1]
        group1.nr_samples = len(indices1)

        group2.X = self.X[indices2]
        group2.Y = self.Y[indices2]
        group2.nr_samples = len(indices2)
        
        
        return group1, group2



    # load the dataset, returns train and test X and y elements
    def load_dataset(self, root='', image_set='train'):
        # load all train
        path = root + image_set + '/X_' + image_set + '.txt'
        X = load_file(path)
        path = root + image_set + '/y_' + image_set + '.txt'
        Y = load_file(path)
        path = root + image_set + '/subject_' + image_set + '.txt'
        subject = load_file(path)
        Y, references = self.mapping_classes(Y)
        
        return X, Y, references, subject

    # mapping Y to sequencial integer value
    def mapping_classes(self, Y):
        # load all train
        classes_id = np.sort(np.unique(Y))
        for i in range(0,len(Y)):
            idx = np.where(classes_id == Y[i])[0]
            Y[i] = idx

        return Y, classes_id

    def add_sample(self, X,Y):
       
        new_X = np.concatenate((self.X, X), axis=0)
        new_Y = np.concatenate((self.Y, Y), axis=0)
        self.X = torch.tensor(new_X, dtype=torch.float64)
        self.Y = torch.tensor(new_Y)
        
   
    def standardize_dataset(self,
        X_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        mode: str = "channel",             # "channel" | "window_samplewise" | "timestep"
        time_cut_ratio: Optional[float] = None,  # only used for "channel"
        eps: float = 1e-8
    ):
        """
        Standardiza dados no formato (N, C, T).
    
        modos:
          - "channel": z-score por CANAL usando stats do TREINO (agrega N e T; opcionalmente só a cauda temporal via time_cut_ratio).
                       Se X_test for dado, aplica as MESMAS stats no teste e retorna X_test_z.
          - "window_samplewise": z-score por CANAL dentro de cada janela (média/desvio da PRÓPRIA janela).
                                 Treino e teste são padronizados amostra a amostra (sem usar stats do treino).
          - "timestep": z-score por CANAL por índice temporal (stats do TREINO para cada t=0..T-1 e aplicadas no teste).
                        Requer T (e C) fixos e alinhados entre treino e teste.                        
          - "timestep_pooled" computes z-score  per timestep, pooled across channels and samples
              
        """
        assert X_train.ndim == 3, "Esperado shape (N, C, T)."
        # match original precision
        X_train = X_train.astype(np.float32, copy=False)
        N, C, T = X_train.shape
    
        if (X_test is not None):
            X_test = X_test.astype(np.float32, copy=False)
    
        if mode == "channel":
            # select tail slice if time_cut_ratio is given (matches original)
            if time_cut_ratio is not None:
                assert 0 < time_cut_ratio <= 1.0
                cut = max(1, int(round(T * time_cut_ratio)))
                T_sel = slice(T - cut, T)
            else:
                T_sel = slice(0, T)
    
            mu = X_train[:, :, T_sel].mean(axis=(0, 2), keepdims=True)       # (1, C, 1)
            sd = X_train[:, :, T_sel].std(axis=(0, 2), keepdims=True) + eps  # (1, C, 1)
    
            X_train_std = (X_train - mu) / sd
            if X_test is None:
                return (X_train_std, mu, sd)
            else:
                X_test_std = (X_test - mu) / sd
                return (X_test_std, mu, sd)
    
        elif mode == "window_samplewise":
            def _win_z(X):
                mu = X.mean(axis=2, keepdims=True)         # (N, C, 1)
                sd = X.std(axis=2, keepdims=True) + eps
                return (X - mu) / sd, mu, sd
    
            X_train_std, mu_tr, sd_tr = _win_z(X_train)
            if X_test is None:
                # return train’s standardized data & its per-window μ/σ
                return (X_train_std, mu_tr, sd_tr)
            else:
                # test is standardized using its own per-window μ/σ (matches original intent)
                X_test_std, mu_te, sd_te = _win_z(X_test)
                return (X_test_std, mu_te, sd_te)
    
        elif mode == "timestep":
            # stats per (channel, timestep) computed on TRAIN; applied to TEST
            mu = X_train.mean(axis=0, keepdims=True)                   # (1, C, T)
            sd = X_train.std(axis=0, keepdims=True) + eps              # (1, C, T)
    
            X_train_std = (X_train - mu) / sd
            if X_test is None:
                return (X_train_std, mu, sd)
            else:
                assert X_test.shape[1] == C, "Para 'timestep', C do teste deve ser igual ao do treino."
                assert X_test.shape[2] == T, "Para 'timestep', T do teste deve ser igual ao do treino."
                X_test_std = (X_test - mu) / sd
                return (X_test_std, mu, sd)
            
        elif mode == "timestep_pooled":  
            # pooled across samples and channels, per timestep
            mu = X_train.mean(axis=(0, 1), keepdims=True)                  # (1,1,T)
            sd = X_train.std(axis=(0, 1), keepdims=True) + eps             # (1,1,T)
        
            X_train_std = (X_train - mu) / sd
            if X_test is None:
                return (X_train_std, mu, sd)
            else:
                X_test_std = (X_test - mu) / sd
                return (X_test_std, mu, sd)                 
        else:
            raise ValueError("mode deve ser 'channel', 'window_samplewise' ou 'timestep'.")
             
            
