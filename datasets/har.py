import os
import copy
import torch
from torch.utils.data import DataLoader, Dataset
from utils.helpers import load_file
import numpy as np
from sklearn.preprocessing import StandardScaler
import datasets.datasetfactory as df
from datasets.utils import  DA_Jitter, DA_Scaling, DA_Permutation, DA_MagWarp, DA_TimeWarp

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
                    batch_size=128, is_standardized=True, subject=all, dataloader=True, data_augmentation=None,  **kwargs):
   
    pin_memory = pin_memory and torch.cuda.is_available  # only pin if GPU available
    #pin_memory = False
    Dataset = get_dataset(dataset)
    if root is None:
        dataset = Dataset(dataset = dataset,
                          is_train=is_train,
                          root = dataset_path,
                          is_standardized=is_standardized)
    else:
        dataset = Dataset(dataset = dataset,
                          root=root,
                          is_train=is_train,
                          is_standardized=is_standardized)

    if data_augmentation is not None:
       dataset = augmentation(dataset, data_augmentation)

    if dataloader:
        return DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=num_workers,
                          pin_memory=pin_memory,
                          **kwargs)
    else:
        return dataset

def augmentation(dataset, data_augmentation):
    
    data = copy.deepcopy(dataset)

    labels_tmp = dataset.Y
    if 'Jitter' in data_augmentation: # Jitter
         print('JITTER')
         dataset_tmp = DA_Jitter(data.X, sigma=0.05)
         dataset.add_sample(dataset_tmp,data.Y)
         print('len data.X : ', len(data.X))
         print('len data.Y : ', len(data.Y))
         print('len dataset_tmp : ', len(dataset_tmp))
         print('len dataset : ', len(dataset))
    if 'Scale' in  data_augmentation:   # Scale
         print('SCALE')
         dataset_tmp = DA_Scaling(data.X, sigma=0.1)
         dataset.add_sample(dataset_tmp,data.Y) 
         print('len data.X : ', len(data.X))
         print('len data.Y : ', len(data.Y))
         print('len dataset_tmp : ', len(dataset_tmp))
         print('len dataset : ', len(dataset))
    if 'Perm' in data_augmentation:   # Permutation
         print('PERM')
         dataset_tmp, labels_tmp = DA_Permutation(data.X,data.Y, nPerm=4, minSegLength=10)           
         dataset.add_sample(dataset_tmp,labels_tmp) 
         print('len data.X : ', len(data.X))
         print('len data.Y : ', len(data.Y))
         print('len labels_tmp : ', len(labels_tmp))
         print('len dataset_tmp : ', len(dataset_tmp))
         print('len dataset : ', len(dataset))
    if 'TimeW' in data_augmentation: # TimeWarping  
         print('TIMEWARPING')
         dataset_tmp = DA_TimeWarp(data.X, sigma=0.2, knot=4)           
         dataset.add_sample(dataset_tmp,data.Y)   
         print('len data.X : ', len(data.X))
         print('len data.Y : ', len(data.Y))
         print('len dataset_tmp : ', len(dataset_tmp))
         print('len dataset : ', len(dataset))
         dataset_tmp = DA_TimeWarp(dataset.X, sigma=0.2, knot=4)           
         dataset.add_sample(dataset_tmp,dataset.Y) 
    if 'MagW' in data_augmentation: # Magnetude Warping 
         print('MAGWARPING')
         dataset_tmp = DA_MagWarp(data.X, sigma=0.2, knot=4)           
         dataset.add_sample(dataset_tmp,data.Y) 
         print('len data.X : ', len(data.X))
         print('len data.Y : ', len(data.Y))
         print('len dataset_tmp : ', len(dataset_tmp))
         print('len dataset : ', len(dataset))
    
    return dataset       

class HumanActivityRecognition(Dataset):
    """Human activity recognition dataset"""

    def __init__(self, dataset, 
                 root=os.path.join(DIR, '../pampa2/'),
                 is_train=True,
                 is_standardized=False):
        """
        Parameters
        ----------

        root : string
            Path to the csv file with annotations.
        is_train : bool
            Chooses train or test set
        is_standardized : bool
            Chooses whether data is standardized
        """
        config = get_dataset_param(dataset)
        self.sensores = config["sensors"]
        self.time_window = config["time_window"]
        self.freq = config["freq"]
        self.data_points = int(self.time_window * self.freq)
        self.data_size = (self.sensores,int(self.data_points))
        self.labels_id = config["labels_id"]
        
       
        if root is None:
            root = config["path"]
            self.dataset_path = config["path"]
        
        if is_train:
            image_set = 'train'
        else:
            image_set = 'test'
    
        if is_standardized:

            data_train = self.load_dataset(root, 'train')
            if image_set == 'train':   

                X = data_train[0].reshape(len(data_train[0]), self.sensores, int(self.data_points))
                X = self.standardize_data(X)
                Y = data_train[1]
                classes_id = data_train[2]
                subject_id = data_train[3]
            elif image_set == 'test':
                print("Loading Human Activity Recognition test dataset ...")
                data_test = self.load_dataset(root, 'test')
                X_train = data_train[0].reshape(len(data_train[0]), self.sensores, int(self.data_points))
                X_test = data_test[0].reshape(len(data_test[0]), self.sensores, int(self.data_points))
                X =  self.standardize_data(X_train, X_test)
                Y = data_test[1]
                classes_id = data_test[2]
                subject_id = data_test[3]
        else:
            print("Loading Human Activity Recognition %s dataset ..." % image_set)
            data = self.load_dataset(root, image_set)
            X = torch.from_numpy(data[0]).view(len(data[0]), self.sensores, int(self.data_points))
            Y = torch.from_numpy(data[1]).flatten().long()
            classes_id = data[2]
            subject_id = data[3]
        
        '''
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.X = torch.from_numpy(X).to(self.device)  # Move to GPU
        self.Y = torch.from_numpy(Y).flatten().long().to(self.device)  # Move to GPU
        '''
        
        
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y).flatten().long()
        
        
        
        self.classes_id = classes_id
        self.nr_classes = len(classes_id)
        self.labels = np.arange(0,len(classes_id))
        self.subject_id = [str(i) for i in np.unique(subject_id)]
        
        
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



    def standardize_timestep_pooled(X_train, X_test=None, use_nonoverlap=True, eps=1e-8):
        # X: (N,C,T)
        if use_nonoverlap:
            X_fit = X_train[::2]        # approx non-overlapping windows
        else:
            X_fit = X_train
        mu = X_fit.mean(axis=(0,1), keepdims=True)           # (1,1,T)
        sd = X_fit.std(axis=(0,1), keepdims=True) + eps      # (1,1,T)
        def tr(X): return (X - mu) / sd
        return (tr(X_train) if X_test is None else tr(X_test), mu, sd)

        #return Y, classes_id
    # standardize data
    def standardize_data(self, X_train, X_test=None):
        """
        Standardizes the dataset

        If X_train is only passed, returns standardized X_train

        If X_train and X_test are passed, returns standardized X_test
        
        Shapes:
            X_*: (N, C, T) where
        N = windows, C = channels, T = timesteps
        -------
        """
        # raise Exception("need to standardize the test set with the mean and stddev of the train set!!!!!!!")
        # remove overlap
        
        # BUG
        #cut = int(X_train.shape[1] / 2)
        #longX_train = X_train[:, -cut:, :]   # BUG
        #longX_train = X_train[:, :, -cut]
                
        # flatten windows
        #longX_train = longX_train.reshape((longX_train.shape[0] * longX_train.shape[1], longX_train.shape[2]))
        # flatten train and test
        #flatX_train = X_train.reshape((X_train.shape[0] * X_train.shape[1], X_train.shape[2]))
        
        # standardize
        #s = StandardScaler()
        # fit on training data
        #s.fit(longX_train)
        
        # FIX
                
        # X: (N, C, T) - N time window, C - sensors, T data points
        
        if X_train.ndim != 3:
                raise ValueError(f"X_train must be (N, C, T); got {X_train.shape}")      
        
        
        N, C, T = X_train.shape
        
        ''' OPTION
        # e.g., keep ~25% of overlapping windows
        keep = np.arange(0, X_train.shape[0], 4)  # or use np.random.choice(...)
        fitX = X_train[keep, :, :]
        fit_flat = fitX.reshape(-1, T)
        
        # use ALL (including overlapping) windows for fitting
        fitX = X_train                            # shape (N, C, T)
        fit_flat = fitX.reshape(-1, T)            # rows = N*C, cols = T (features = timesteps)
        '''
        
        # use non-overlapping windows for fitting (approx: every other window)
        fitX = X_train[::2, :, :]                 # shape (~N/2, C, T)
        fit_flat = fitX.reshape(-1, T)           # rows = Nfit*C, cols = T (features = timesteps)
        
        scaler = StandardScaler()
        scaler.fit(fit_flat)                # features = timesteps (T)  
        
        # apply to training and test data
        if X_test is not None:
            # Standardize TEST: pool channels; standardize per timestep
            flat_te = X_test.reshape(-1, T)      # (N2*C, T)
            flat_te = scaler.transform(flat_te)
            X_test_std = flat_te.reshape(X_test.shape)
            return X_test_std
        else:
            # Standardize TRAIN: apply to ALL train windows (not just the fit subset)
            flat_tr = X_train.reshape(-1, T)     # (N*C, T)
            flat_tr = scaler.transform(flat_tr)
            X_train_std = flat_tr.reshape(X_train.shape)
            return X_train_std