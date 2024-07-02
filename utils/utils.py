import errno
import hashlib
import os
import os.path
import random
from collections import namedtuple
import logging
logger = logging.getLogger('experiment')
from torch.nn import functional as F
import numpy as np
import copy
import pandas as pd
from sklearn.metrics import confusion_matrix
import utils.stats as st
import sys
import tensorflow as tf
import json
import gzip


transition = namedtuple('transition', 'state, next_state, action, reward, is_terminal')
import torch


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def codify_obj(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, ReplayBuffer):
        return obj.__dict__
    raise TypeError(f"{type(obj)} não é serializável")


def serialize_tensors(dictionary):
    serialized_dict = {}
    for key, tensor in dictionary.items():
        serialized_dict[key] = tensor.numpy().tolist()
    return serialized_dict

    
def save_json_gz(obj, filename):
    obj_json = json.dumps(obj, default=codify_obj)

    with gzip.open(filename, 'wb') as file:
        file.write(obj_json.encode('utf-8'))
        
        
        
def load_json_gz(filename):
    with gzip.open(filename, 'rb') as file:
        obj_json_load = file.read()

    return json.loads(obj_json_load.decode('utf-8'))
    
    
    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def freeze_layers(layers_to_freeze, maml):

    for name, param in maml.named_parameters():
        param.learn = True

    for name, param in maml.net.named_parameters():
        param.learn = True

    frozen_layers = []
    for temp in range(layers_to_freeze * 2):
        frozen_layers.append("net.vars." + str(temp))

    for name, param in maml.named_parameters():
        if name in frozen_layers:
            logger.info("RLN layer %s", str(name))
            param.learn = False

    list_of_names = list(filter(lambda x: x[1].learn, maml.named_parameters()))

    for a in list_of_names:
        logger.info("TLN layer = %s", a[0])

def log_accuracy(maml, my_experiment, iterator_test, device, writer, step):
    correct = 0
    torch.save(maml.net, my_experiment.path + "learner.model")
    for img, target in iterator_test:

        with torch.no_grad():
            img = img.to(device)
            target = target.to(device)
            logits_q = maml.net(img, vars=None)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct += torch.eq(pred_q, target).sum().item() / len(img)
    writer.add_scalar('/metatrain/test/classifier/accuracy', correct / len(iterator_test), step)
    logger.info("Test Accuracy = %s", str(correct / len(iterator_test)))

def log_accuracy_har(maml, my_experiment, iterator_test, device, writer, step, labels, id_task, args):

    confusion_pred = []
    confusion_act = []

    for X, Y in iterator_test:

        with torch.no_grad():
            X = X.to(device)
            Y = Y.to(device)
            logits_q = maml.net(X, vars=None)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            confusion_pred += pred_q.tolist()
            confusion_act += Y.tolist()
            
    if args['dataset'] == "ucihar":
       labels_iteration = sorted(np.unique([confusion_pred, confusion_act]))
       confusion_mat = pd.DataFrame(confusion_matrix(y_true=confusion_act, y_pred=confusion_pred,labels=labels_iteration))
    else:
        confusion_mat = pd.DataFrame(confusion_matrix(y_true=confusion_act, y_pred=confusion_pred))
        labels_iteration = labels = np.array(range(confusion_mat.shape[0]))
       
    stats = st.Stats(confusion_mat, labels_iteration, step)
    
    writer.add_scalar('/metatrain/test/classifier/accuracy', stats.accuracy, step)
    logger.info(id_task + " Accuracy = %s", str(stats.accuracy))
    logger.info(id_task + " Weighted Macro F1-Score = %s", str(stats.f1_scores_weighted))
    
    
    results = {
       "Accuracy": stats.accuracy,
       "Precision per class " : stats.precisions.tolist(),
       "F1-score  macro": stats.macro_f1,
       "F1-score micro": stats.micro_f1,
       "F1-score weighted": stats.f1_scores_weighted,
       "Labels": list(map(str,stats.labels)),
       "F1-scores per class" : stats.f1_scores.tolist(),
       "Confusion matrix " : confusion_mat.values.tolist()
       }
 
    
    return stats, results

def log_accuracy_har_v2(maml, my_experiment, iterator_test, device, writer, step, labels, id_task, args):

    confusion_pred = []
    confusion_act = []
    print('id_task ', id_task)

    for X, Y in iterator_test:
        print('iterator test inside log_accuracy_har_v2 ', Y)

        with torch.no_grad():
            X = X.to(device)
            Y = Y.to(device)
            logits_q = maml(X, vars=None)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            confusion_pred += pred_q.tolist()
            confusion_act += Y.tolist()
         
    labels_pred = sorted(np.unique([confusion_pred, confusion_act]))
    labels_iteration_pred = np.array(list(map(str,  labels_pred)))   
    confusion_mat = pd.DataFrame(confusion_matrix(y_true=confusion_act, y_pred=confusion_pred,labels=labels_pred),index = labels_iteration_pred,columns=labels_iteration_pred)  
    print('labels_pred' , labels_pred)
    print('labels_iteration_pred' , labels_iteration_pred)
    print('before stats')
    stats = st.Stats(confusion_mat, step)
    print('after stats')
    writer.add_scalar('/metatrain/test/classifier/accuracy', stats.accuracy, step)
    logger.info(id_task + " Accuracy = %s", str(stats.accuracy))
    logger.info(id_task + " Weighted Macro F1-Score = %s", str(stats.f1_scores_weighted))
   
  
    return stats
 
    

    
# convert the labels used in learning to the original dataset class label
# class_ref : labels used in learning
# class_id: list of original dataset learning
# class_idx: list of all labels used in learning
def get_class_labels(class_ref, class_id, class_idx):
    labels = []       
    for i in range(0,len(class_ref)):
         pos = class_idx[0].index(class_ref[i])
         s = "-"
         s = s.join([class_ref[i],class_id[0][pos]])
         labels.append(s)
    
    return labels

def prepare_json_dataset(dataloader):
    class_description = []
    labels = dataloader.get_class_labels()
    for i in range(0,len(labels)):
        class_description.append(dataloader.get_class_description(labels[i]))
    results = {
       "Class labels": [str(num) for num in dataloader.get_class_labels()], # standarized label
       "Class id": [str(num) for num in dataloader.get_classes_id()], # reference in dataset
       "Class description": class_description
       }
       
    return results
 

def prepare_json_stats(stats):
    
    results = {
       "Step": stats.step,
       "Accuracy": stats.accuracy,
       "Precision": stats.macro_precision,
       "Precision per class" : stats.precisions.tolist(),
       "F1-score  macro": stats.macro_f1,
       "F1-score micro": stats.micro_f1,
       "F1-score weighted": stats.f1_scores_weighted,
       "Labels": stats.labels.tolist(),
       "F1-scores per class" : stats.f1_scores.tolist()
       }
    return results


def prepare_json_stats_baseline(stats):
    
    
    results = {
       "Step": [row[0] for row in stats],
       "Loss": [row[1] for row in stats],
       "Accuracy": [row[2] for row in stats]
       }
    return results

def prepare_json_forgetting(forgetting):
    
    results = {
           "Forgetting": average_forgetting(forgetting),
           "Forgetting per iteration" : average_forgetting(forgetting, category='Iteration').tolist(),
           "Forgetting per class" : average_forgetting(forgetting, category='Class').tolist(),
           }
    return results

def prepare_json_backwardTransfer(backward_transfer):
    
    results = {
           "Backward Transfer": average_forgetting(backward_transfer),
           "Backward Transfer per iteration" : average_forgetting(backward_transfer, category='Iteration').tolist(),
           "Backward Transfer per class" : average_forgetting(backward_transfer, category='Class').tolist(),
           }
    return results

def prepare_json_backwardTransfer_v2(backward_transfer,backward_transfer_t0,backward_transfer_max):
    
    results = {
           "Backward Transfer": average_forgetting(backward_transfer),
           "Backward Transfer per iteration" : average_forgetting(backward_transfer, category='Iteration').tolist(),
           "Backward Transfer per class" : average_forgetting(backward_transfer, category='Class').tolist(),
           "Backward Transfer t0": average_forgetting(backward_transfer_t0),
           "Backward Transfer per iteration t0" : average_forgetting(backward_transfer_t0, category='Iteration').tolist(),
           "Backward Transfer per class t0" : average_forgetting(backward_transfer_t0, category='Class').tolist(),
           "Backward Transfer max": average_forgetting(backward_transfer_max),
           "Backward Transfer per iteration max" : average_forgetting(backward_transfer_max, category='Iteration').tolist(),
           "Backward Transfer per class max" : average_forgetting(backward_transfer_max, category='Class').tolist(),         
          
           }
    return results

def log_accuracy_har2(maml, my_experiment, iterator_test, device, writer, step, labels, id_task, args):

    
    confusion_pred = []
    confusion_act = []

    for X, Y in iterator_test:
        with torch.no_grad():
             X = X.to(device)
             Y = Y.to(device)
             logits_q = maml.net(X, vars=None)
             pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
             confusion_pred += pred_q.tolist()
             confusion_act += Y.tolist()
             
             
    labels_iteration = sorted(np.unique([confusion_pred, confusion_act]))
  
    confusion_mat = pd.DataFrame(confusion_matrix(y_true=confusion_act, y_pred=confusion_pred,labels=labels_iteration)) 

    stats = st.Stats(confusion_mat, labels_iteration, step)
    forgetting = compute_forgetting(stats, labels, args)
    if forgetting != -99:
        avg_forgetting = average_forgetting(forgetting, step=True)
        forgetting_original = backward_transfer(stats, labels, args)
        avg_forgetting_original = average_forgetting(forgetting_original, step=True)
    else:
        avg_forgetting = forgetting_original =  avg_forgetting_original = -99
    
    
    writer.add_scalar('/metatrain/test/classifier/accuracy', stats.accuracy, step)
    logger.info(id_task + " Accuracy = %s", str(stats.accuracy))
    logger.info(id_task + " Weighted Macro F1-Score = %s", str(stats.f1_scores_weighted))
    logger.info(id_task + " Average Forgetting = %s", str(avg_forgetting))
    logger.info(id_task + " Average Forgetting Original = %s", str(avg_forgetting_original))
    
    
    results = {
       "Accuracy": stats.accuracy,
       "Precision per class " : stats.precisions.tolist(),
       "F1-score  macro": stats.macro_f1,
       "F1-score micro": stats.micro_f1,
       "F1-score weighted": stats.f1_scores_weighted,
       "Labels": list(map(str,stats.labels)),
       "F1-scores per class" : stats.f1_scores.tolist(),
       "Forgetting " : avg_forgetting,
       "Forgetting per class" : forgetting,
       "Forgetting original " : avg_forgetting_original,
       "Forgetting original per class" : forgetting_original
       }
 
    
    return results


class replay_buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []

    def add(self, *args):
        # Append when the buffer is not full but overwrite when the buffer is full
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition(*args))
        else:
            self.buffer[self.location] = transition(*args)

        # Increment the buffer location
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def sample_trajectory(self, batch_size):
        initial_index = random.randint(0, len(self.buffer) - batch_size)
        return self.buffer[initial_index: initial_index + batch_size]


class ReservoirSampler:
    def __init__(self, windows, buffer_size=5000):
        self.buffer = []
        self.location = 0
        self.buffer_size = buffer_size
        self.window = windows
        self.total_additions = 0

    def add(self, *args):
        self.total_additions += 1
        stuff_to_add = transition(*args)

        M = len(self.buffer)
        if M < self.buffer_size:
            self.buffer.append(stuff_to_add)
        else:
            i = random.randint(0, min(self.total_additions, self.window))
            if i < self.buffer_size:
                self.buffer[i] = stuff_to_add
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def sample_trajectory(self, batch_size):
        initial_index = random.randint(0, len(self.buffer) - batch_size)
        return self.buffer[initial_index: initial_index + batch_size]


def iterator_sorter(trainset, no_sort=True, random=True, pairs=False, classes=10):
    if no_sort:
        return trainset

    order = list(range(len(trainset.data)))
    np.random.shuffle(order)

    trainset.data = trainset.data[order]
    trainset.targets = np.array(trainset.targets)
    trainset.targets = trainset.targets[order]

    sorting_labels = np.copy(trainset.targets)
    sorting_keys = list(range(20, 20 + classes))
    if random:
        if not pairs:
            np.random.shuffle(sorting_keys)

    print("Order = ", [x - 20 for x in sorting_keys])
    for numb, key in enumerate(sorting_keys):
        if pairs:
            np.place(sorting_labels, sorting_labels == numb, key - (key % 2))
        else:
            np.place(sorting_labels, sorting_labels == numb, key)

    indices = np.argsort(sorting_labels)
    # print(indices)

    trainset.data = trainset.data[indices]
    trainset.targets = np.array(trainset.targets)
    trainset.targets = trainset.targets[indices]
    # print(trainset.targets)
    # print(trainset.targets )

    return trainset


def iterator_sorter_omni(trainset, no_sort=True, random=True, pairs=False, classes=10):
    return trainset


def remove_classes(trainset, to_keep):

    trainset.targets = np.array(trainset.targets)

    indices = np.zeros_like(trainset.targets)
    for a in to_keep:
        indices = indices + (trainset.targets == a).astype(int)
    indices = np.nonzero(indices)

    trainset.data = trainset.data[indices]
    trainset.targets = np.array(trainset.targets)
    trainset.targets = trainset.targets[indices]

    return trainset


def remove_classes_omni(trainset, to_keep):

    trainset = copy.deepcopy(trainset)
    # trainset.data = trainset.data[order]
    trainset.targets = np.array(trainset.targets)
    # trainset.targets = trainset.targets[order]

    indices = np.zeros_like(trainset.targets)
    for a in to_keep:
        indices = indices + (trainset.targets == a).astype(int)
    indices = np.nonzero(indices)
    trainset.data = [trainset.data[i] for i in indices[0]]
    # trainset.data = trainset.data[indices]
    trainset.targets = np.array(trainset.targets)
    trainset.targets = trainset.targets[indices]

    return trainset


def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)


def list_dir(root, prefix=False):
    """List all directories at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


def resize_image(img, factor):
    '''

    :param img:
    :param factor:
    :return:
    '''
    img2 = np.zeros(np.array(img.shape) * factor)

    for a in range(0, img.shape[0]):
        for b in range(0, img.shape[1]):
            img2[a * factor:(a + 1) * factor, b * factor:(b + 1) * factor] = img[a, b]
    return img2



def get_run(arg_dict, rank=0):
    # print(arg_dict)
    combinations =[]

    if isinstance(arg_dict["seed"], list):
        combinations.append(len(arg_dict["seed"]))


    for key in arg_dict.keys():
        if isinstance(arg_dict[key], list) and not key=="seed":
            combinations.append(len(arg_dict[key]))

    total_combinations = np.prod(combinations)
    selected_combinations = []
    for base in combinations:
        selected_combinations.append(rank%base)
        rank = int(rank/base)

    counter=0
    result_dict = {}

    result_dict["seed"] = arg_dict["seed"]
    if isinstance(arg_dict["seed"], list):
        result_dict["seed"] = arg_dict["seed"][selected_combinations[0]]
        counter += 1
    #

    for key in arg_dict.keys():
        if key !="seed":
            result_dict[key] = arg_dict[key]
            if isinstance(arg_dict[key], list):
                result_dict[key] = arg_dict[key][selected_combinations[counter]]
                counter+=1

    logger.info("Parameters %s", str(result_dict))
    # 0/0
    return result_dict

import torch
#
def construct_set(iterators, sampler, steps, shuffle=True):
    x_traj = []
    y_traj = []

    x_rand = []
    y_rand = []


    id_map = list(range(sampler.capacity - 1))
    if shuffle:
        random.shuffle(id_map)

    for id, it1 in enumerate(iterators):
        id_mapped = id_map[id]
        for inner in range(steps):
            x, y = sampler.sample_batch(it1, id_mapped, 10)
            x_traj.append(x)
            y_traj.append(y)
        #
        x, y = sampler.sample_batch(it1, id_mapped, 10)
        x_rand.append(x)
        y_rand.append(y)

    x_rand = torch.stack([torch.cat(x_rand)])
    y_rand = torch.stack([torch.cat(y_rand)])

    x_traj = torch.stack(x_traj)
    y_traj = torch.stack(y_traj)


    return x_traj, y_traj, x_rand, y_rand


# load metadata file to sample ucihar meta-test 
def load_dataset_metadata(root, group, prefix):
        
    filename =  root + group + '/' + prefix +'_' + group + '.txt'
    
    metadata = pd.read_csv(filename, header=None, names=['Data'])

    return metadata


def sample_metatest_data(data, target, root, group, task):

    data = copy.deepcopy(data)
    metadata =  load_dataset_metadata(root, group, task)
    
    if not (all(isinstance(val, int) for val in target)):
       target = [int(x) for x in target]  # convert to list of integer
    index = metadata.index[metadata['Data'].isin(target)]

    data.X = data.X[index]
    data.Y = data.Y[index]
    return data



def sample_subject(data, target, root, group):

    class_id = []
    
    for i in range(0,len(target)):
        class_id.append(data.get_class_id(int(target[i]))[0])

    metadata_subject =  load_dataset_metadata(root, group, 'subject')
    metadata_classes =  load_dataset_metadata(root, group, 'y')
    
    subjects = np.unique(metadata_subject['Data'])
    
    subjects_candidate = []
    
    for subject in subjects:
        pos = np.where(metadata_subject['Data'] == subject)[0]
        sample_data = pd.DataFrame(metadata_classes['Data'].iloc[pos])
        value_counts = sample_data['Data'].value_counts()
        
        intersection = set(class_id).intersection(value_counts.index)

        if intersection == set(class_id):
           subjects_candidate.append(str(subject))
        
    
    return subjects_candidate



def remove_classes_ucihar(data, target):

    data = copy.deepcopy(data)
    data.Y = np.array(data.Y)
    index = np.zeros_like(data.Y)
    for x in target:
        index = index + (data.Y == x).astype(int)
    index = np.nonzero(index)
    data.X = data.X[index]
    data.Y = torch.from_numpy(data.Y[index])

    return data

def compute_forgetting(stats,  classes_to_keep, args):
   
    if (type(stats) is list):
        mat = np.full((len(stats),len(classes_to_keep)),np.NINF,dtype=np.float64)
    else:
        print('Not possible to compute forgetting - only one instance!')
        return np.NINF
    count = 0    
    index_classes_to_keep = classes_to_keep
    for data in stats:
        index_classes = data.labels
        for idx in range(len(data.f1_scores)):
            if data.labels[idx] in classes_to_keep:                     
                pos = index_classes.get_loc(data.labels[idx])
                col = index_classes_to_keep.index(data.labels[idx])
                mat[count,col] = data.f1_scores[pos]
        count += 1       
            
    forgetting =  np.zeros(( (mat.shape[0]-1), len(classes_to_keep)) )   
            
    for c in range(mat.shape[1]):
        idx = 0
        for l in range(1,mat.shape[0]):
            max_f1 =  mat[0:l,c].max()
            if mat[l,c] == np.NINF:
                forgetting[idx,c] = np.NINF if max_f1 == np.NINF else 0
            else:
                forgetting[idx,c] =  1 - mat[l,c]/max_f1 if max_f1 > 0 else 0
            idx +=1

    return forgetting     

 


def backward_transfer(stats,  classes_to_keep, args):
   
   
   if (type(stats) is list):
       mat = np.full((len(stats),len(classes_to_keep)),np.NINF,dtype=np.float64)
   else:
       print('Not possible to compute forgetting - only one instance!')
       return np.NINF
   count = 0    
   index_classes_to_keep = classes_to_keep
   for data in stats:
       index_classes = data.labels
       for idx in range(len(data.f1_scores)):
           if data.labels[idx] in classes_to_keep:                     
               pos = index_classes.get_loc(data.labels[idx])
               col = index_classes_to_keep.index(data.labels[idx])
               mat[count,col] = data.f1_scores[pos]
       count += 1       
    
   #forgetting =  np.zeros(( (mat.shape[0]-1), len(classes_to_keep)) ) 
   forgetting =  np.full(( (mat.shape[0]-1), len(classes_to_keep)),np.NINF,dtype=np.float64)
            
              
   for c in range(mat.shape[1]):
        idx = 0

        for l in range(1,mat.shape[0]):
            sum_forgetting = 0
            count = 0
            if mat[l,c] != np.NINF:
               t = l - 1
               while t >= 0:
                   if mat[t,c] != np.NINF:
                      sum_forgetting +=  mat[l,c] - mat[t,c]
                      count += 1 
                   t -= 1
               if count == 0:
                   forgetting[idx,c] = 0
               else:
                   forgetting[idx,c] =  (sum_forgetting/count)

            idx +=1

   return forgetting   

def backward_transfer_v2(stats,  classes_to_keep, args):
   
   
   if (type(stats) is list):
       mat = np.full((len(stats),len(classes_to_keep)),np.NINF,dtype=np.float64)
   else:
       print('Not possible to compute forgetting - only one instance!')
       return np.NINF
   count = 0    
   index_classes_to_keep = classes_to_keep
   for data in stats:
       index_classes = data.labels
       for idx in range(len(data.f1_scores)):
           if data.labels[idx] in classes_to_keep:                     
               pos = index_classes.get_loc(data.labels[idx])
               col = index_classes_to_keep.index(data.labels[idx])
               mat[count,col] = data.f1_scores[pos]
       count += 1       
    
   #forgetting =  np.zeros(( (mat.shape[0]-1), len(classes_to_keep)) )   
   forgetting =  np.full(( (mat.shape[0]-1), len(classes_to_keep)),np.NINF,dtype=np.float64)         
              
   for c in range(mat.shape[1]):
        idx = 0
        
        for l in range(1,mat.shape[0]):
            sum_forgetting = 0
            count = 0
            if mat[l,c] != np.NINF:
               t = l - 1
               ref = mat[l,c]
               while t >= 0: 
                  if mat[t,c] != np.NINF: 
                      sum_forgetting +=   ref - mat[t,c]
                      ref = mat[t,c]
                      count += 1 
                   
                  t -= 1
               if count == 0:
                  forgetting[idx,c] = 0
               else:
                  forgetting[idx,c] =  (sum_forgetting/count) 
            idx +=1

   return forgetting   

def backward_transfer_prior(stats,  classes_to_keep, args):
   
   
   if (type(stats) is list):
       mat = np.full((len(stats),len(classes_to_keep)),np.NINF,dtype=np.float64)
   else:
       print('Not possible to compute forgetting - only one instance!')
       return np.NINF
   count = 0    
   index_classes_to_keep = classes_to_keep
   for data in stats:
       index_classes = data.labels
       for idx in range(len(data.f1_scores)):
           if data.labels[idx] in classes_to_keep:                     
               pos = index_classes.get_loc(data.labels[idx])
               col = index_classes_to_keep.index(data.labels[idx])
               mat[count,col] = data.f1_scores[pos]
       count += 1       
    
   #forgetting =  np.zeros(( (mat.shape[0]-1), len(classes_to_keep)) )   
   forgetting =  np.full(( (mat.shape[0]-1), len(classes_to_keep)),np.NINF,dtype=np.float64)         
              
   for c in range(mat.shape[1]):
        idx = 0

        for l in range(1,mat.shape[0]):
            forg = 0
            if mat[l,c] != np.NINF:
               t = l - 1
               while True:
                  if mat[t,c] != np.NINF and  t>=0:
                     forg = mat[l,c] - mat[t,c]
                     break
                  t -= 1
               forgetting[idx,c] = forg
            idx += 1
  
   return forgetting   

def backward_transfer_t0(stats,  classes_to_keep, args):
   
   
   if (type(stats) is list):
       mat = np.full((len(stats),len(classes_to_keep)),np.NINF,dtype=np.float64)
   else:
       print('Not possible to compute forgetting - only one instance!')
       return np.NINF
   count = 0    
   index_classes_to_keep = classes_to_keep
   for data in stats:
       index_classes = data.labels
       for idx in range(len(data.f1_scores)):
           if data.labels[idx] in classes_to_keep:                     
               pos = index_classes.get_loc(data.labels[idx])
               col = index_classes_to_keep.index(data.labels[idx])
               mat[count,col] = data.f1_scores[pos]
       count += 1       
    
   #forgetting =  np.zeros(( (mat.shape[0]-1), len(classes_to_keep)) )   
   forgetting =  np.full(( (mat.shape[0]-1), len(classes_to_keep)),np.NINF,dtype=np.float64)
  
   for c in range(mat.shape[1]):
        idx = 0
        notInf = np.where(np.isinf(mat[:,c]), False, True)
        index = np.where(notInf)
        if len(index[0]) != 0:
           first_notInf =  index[0][0]
              
           for l in range(1,mat.shape[0]):
              if mat[l,c] != np.NINF:
                 forgetting[idx,c] = mat[l,c] - mat[first_notInf,c]
              idx += 1

   return forgetting   

def backward_transfer_max(stats,  classes_to_keep, args):
   
   
   if (type(stats) is list):
       mat = np.full((len(stats),len(classes_to_keep)),np.NINF,dtype=np.float64)
   else:
       print('Not possible to compute forgetting - only one instance!')
       return np.NINF
   count = 0    
   index_classes_to_keep = classes_to_keep
   for data in stats:
       index_classes = data.labels
       for idx in range(len(data.f1_scores)):
           if data.labels[idx] in classes_to_keep:                     
               pos = index_classes.get_loc(data.labels[idx])
               col = index_classes_to_keep.index(data.labels[idx])
               mat[count,col] = data.f1_scores[pos]
       count += 1       
    
   #forgetting =  np.zeros(( (mat.shape[0]-1), len(classes_to_keep)) )   
   forgetting =  np.full(( (mat.shape[0]-1), len(classes_to_keep)),np.NINF,dtype=np.float64)
  
   for c in range(mat.shape[1]):
          idx = 0
          for l in range(1,mat.shape[0]):
              if mat[l,c] != np.NINF:
                # if l == 1 and mat[0,c] is not np.NINF:
                #    forgetting[idx,c] = mat[l,c] - mat[0,c]
                # else:   
                notInf = np.where(np.isinf(mat[0:l,c]), False, True)
                index = np.where(notInf) 
                if len(index[0]) != 0:
                   forgetting[idx,c] = mat[l,c] - max(mat[index[0][:],c]) 
                else:
                   forgetting[idx,c] = 0 
              idx += 1      
           

   return forgetting   

def average_forgetting(forgetting, category='Global'):
    
    if category == 'Iteration':
        avg =  np.zeros(len(forgetting))  
        for idx in range(0,len(forgetting)):
            if (forgetting[idx,:] != np.NINF).any():
                data = forgetting[idx,:] 
                avg[idx] = np.mean(data[data != np.NINF]) 
            else:
                avg[idx] = np.NINF
        return avg
    else: 
        if category == 'Class':
            avg =  np.zeros(forgetting.shape[1])
            for idx in range(0,forgetting.shape[1]):
                if (forgetting[:,idx] != np.NINF).any():
                    data = forgetting[:,idx]
                    avg[idx] = np.mean(data[data != np.NINF]) 
                else:
                    avg[idx] = np.NINF
            return avg
        else:
            return np.mean(forgetting[forgetting != np.NINF])  

    
def average_stats(stats):
  
    results = {
       "Accuracy": np.mean([c.accuracy for c in stats]),
       "Accuracy std": np.std([c.accuracy for c in stats]),
       "F1-score  macro": np.mean([c.macro_f1 for c in stats]),
       "F1-score  macro std": np.std([c.macro_f1 for c in stats]),
       "F1-score weighted": np.mean([c.f1_scores_weighted for c in stats]),
       "F1-score weighted std": np.std([c.f1_scores_weighted for c in stats]),
       "Macro recall": np.mean([c.macro_recall for c in stats]),
       "Macro recall std": np.std([c.macro_recall for c in stats]),
       "Macro precision":np.mean([c.macro_precision for c in stats]),
       "Macro precision std":np.std([c.macro_precision for c in stats])
       }
    return results


def add_replay_buffer(x,y, replay_buffer):
    
   replay = replay_buffer.replay 
   for key in replay.keys():
       for idx in range(0,len(replay[key])):
           replay_expanded = replay[key][idx].unsqueeze(0).expand(1, -1, -1)
           x = torch.cat((x, replay_expanded), dim=0)
           y = torch.cat((y,torch.tensor([key])),dim=0)
        
   return x,y


def add_replay_buffer_dataset(data, replay_buffer):
    
    datasetDst = copy.deepcopy(data)   
    replay = replay_buffer.replay 
    for key in replay.keys():
        for idx in range(0,len(replay[key])):
            replay_expanded = replay[key][idx].unsqueeze(0).expand(1, -1, -1)
            datasetDst.X = torch.cat((datasetDst.X, replay_expanded), dim=0)
            datasetDst.Y = torch.cat((datasetDst.Y,torch.tensor([key])),dim=0)
        
    return datasetDst


def sample_meta_updating(X,Y, classes, num_support, num_query, reset, random): 

    # Sample data for inner and meta updates
    # classes = classes with changing in distribution 
    # X : data 
    # Y: ground truth 

    x_traj, y_traj, x_rand, y_rand = [], [], [], []

    # counts the number of samples of each class
    unique, counts = np.unique(Y, return_counts=True)
    
    # get the minimum number of samples 

    nr_samples  = min([item for (index, item) in enumerate(counts) if item > 1])
    
    # if the number of samples are not enough to accomodate support and query samples
    # they will be equaly split into spt and qry

    if int(nr_samples) < (num_support + num_query):
       num_support = num_query = int(nr_samples/2)
    
    print('num_support ', num_support)
    print('num_samples ', nr_samples)
    for c in classes:
        k = torch.where(Y == c)

        data = k[0].numpy()
        if len(data) == 1:
           x_rand.append(X[data[0]])
           y_rand.append(Y[data[0]])
           continue
        else:
           sample_positions = np.random.choice(len(data),nr_samples,replace=False)
           counter = 0
        
           for j in sample_positions:
      
                if counter < num_support:
                    x_traj.append(X[data[j]])
                    y_traj.append(Y[data[j]])
                    
                else:
                    if counter == (num_query + num_support):
                        break
                    x_rand.append(X[data[j]])
                    y_rand.append(Y[data[j]])
                    
                counter += 1;

    x_traj = torch.stack(x_traj).unsqueeze(1)
    y_traj = torch.stack(y_traj).unsqueeze(1)   
    
    x_rand = torch.stack(x_rand).unsqueeze(0)
    y_rand = torch.stack(y_rand).unsqueeze(0)
  

    return x_traj, y_traj, x_rand, y_rand     

def sample_meta_updating_v2(X,Y, replay, classes, num_support, num_query, reset, random): 

    # Sample data for inner and meta updates
    # classes = classes with changing in distribution 
    # X : data 
    # Y: ground truth 
    
    if replay is not None:
       X, Y = add_replay_buffer(X, Y, replay) 
    x_traj, y_traj, x_rand, y_rand = [], [], [], []
    classes_samples = [] 
    classes_support = []

    for c in classes:
        k = torch.where(Y == c)
        # there is only one sample
        if (len(k[0].numpy()) > 1):
           classes_samples.append(len(k[0].numpy()))
           classes_support.append(c)
    
    #print('classes_support ', classes_support)
    
    if len(classes_support) < 2:
       return x_traj, y_traj, x_rand, y_rand
    
    nr_samples = min(classes_samples)    
    if nr_samples < (num_support + num_query):
       num_support = num_query = int(nr_samples/2)
    
    for c in classes_support:
        k = torch.where(Y == c)

        data = k[0].numpy()
        
        if len(data) == 1:
           x_rand.append(X[data[0]])
           y_rand.append(Y[data[0]])
           continue
        
        else:
           sample_positions = np.random.choice(len(data),nr_samples,replace=False)
           counter = 0
        
           for j in sample_positions:
      
                if counter < num_support:
                    x_traj.append(X[data[j]])
                    y_traj.append(Y[data[j]])
                    
                else:
                    if counter == (num_query + num_support):
                        break
                    x_rand.append(X[data[j]])
                    y_rand.append(Y[data[j]])
                    
                counter += 1;
        
        # select one element of each class that has not being updated to
        # query set
        
    classes_in_streaming = set(np.unique(Y))
    
    classes_not_updated = classes_in_streaming.difference(set(classes_support))
    
    
    for c in  classes_not_updated:
        k = torch.where(Y == c)

        data = k[0].numpy()
        position = np.random.choice(len(data),1,replace=False)
        x_rand.append(X[data[position[0]]])
        y_rand.append(Y[data[position[0]]])
            

    x_traj = torch.stack(x_traj).unsqueeze(1)
    y_traj = torch.stack(y_traj).unsqueeze(1)   
    
    x_rand = torch.stack(x_rand).unsqueeze(0)
    y_rand = torch.stack(y_rand).unsqueeze(0)
  

    return x_traj, y_traj, x_rand, y_rand     


def create_dictionary(buffer):
    replay_dict = {}
    
    for key, value in buffer:
        if key in replay_dict and value not in replay_dict:
            replay_dict[key].append(value)
        else:
            replay_dict[key] = [value]
            
    return replay_dict             


def check_replay_update(updating, device, args):
    
    # find classes with distribution changed according to the replay strategy
    # returns classes with shift in distribution
    positions = np.where(np.array(updating)[:,1] > 0)
    print('updating', updating)
    c = []
    if positions[0].size > 0:
        for i in positions[0]:
             c.append(updating[i][0])
    return c
        


def average_stats_classes(stats, classes_to_keep):
    
    if (type(stats) is list):
        f1_classes =  np.zeros((len(stats),len(classes_to_keep)),dtype=np.float64)
        precision_classes =  np.zeros((len(stats),len(classes_to_keep)),dtype=np.float64)
        f1_weighted_classes =  np.zeros((len(stats),len(classes_to_keep)),dtype=np.float64)
        
    else:
        print('Not possible to compute average - only one instance!')
        return np.NINF
    count = 0    
    index_classes_to_keep = classes_to_keep
    for data in stats:
        index_classes = data.labels
        for idx in range(len(data.f1_scores)):
            
            if data.labels[idx] in classes_to_keep:                     
                pos = index_classes.get_loc(data.labels[idx])
                col = index_classes_to_keep.index(data.labels[idx])
                f1_classes[count,col] =  data.f1_scores[pos]
                precision_classes[count,col] =  data.precisions[pos]
                f1_weighted_classes[count,col] = data.f1_scores_weighted_classes[pos]
                
        count += 1       

    results = {
       "classes": classes_to_keep, 
       "f1_scores average" : np.mean(f1_classes),
       "f1_scores std" : np.std(np.mean(f1_classes, axis=1)),       
       "f1-score weighted average": np.mean(np.sum(f1_weighted_classes,axis=1)),
       "f1-score weighted std": np.std(np.sum(f1_weighted_classes,axis=1)),
       "f1_classes": np.mean(f1_classes, axis=0).tolist(),
       "f1_classes std": np.std(f1_classes, axis=0).tolist(),
       "precision_classes": np.mean(precision_classes,axis=0).tolist(),
       "precision_classes std": np.std(precision_classes,axis=0).tolist(),
       "precision average": np.mean(precision_classes),
       "precision std": np.std(np.mean(precision_classes, axis=1)),
    }
    return results


def mergeDictionary(dict_1, dict_2):
   dict_3 = {**dict_1, **dict_2}
   for key, value in dict_3.items():
       if key in dict_1 and key in dict_2:
               dict_3[key] = [value , dict_1[key]]
   return dict_3


class ReplayBuffer:
    def __init__(self, buffer=None, buffer_size_class=None, strategy=None, center_strategy=None):
        super().__init__()

        self.buffer_size = buffer_size_class
        self.strategy = strategy
        self.center_strategy = center_strategy
        self.replay = None
        self.average = None
        self.size = None

        if buffer is not None and buffer_size_class is not None and strategy is not None and center_strategy is not None:
            replay_dict = create_dictionary(buffer)

            self.replay = {}
            self.average = {}
            self.size = {}

            if self.strategy == 'random':
                self.random_sample(replay_dict)
            elif self.strategy == 'exemplar':
                self.exemplar_sample(replay_dict)
            elif self.strategy == 'herding':
                self.herding_sample(replay_dict)
            else:
                print('strategy not valid')
                sys.exit()

            print('\n strategy:', strategy)
            print('keys ', self.replay.keys())

    def set_values(self, buffer, buffer_size, strategy, center_strategy, average, size):
        
        self.replay = buffer
        self.buffer_size = buffer_size
        self.strategy = strategy
        self.center_strategy = center_strategy
        self.average = average
        self.size = size 


    def serialize_tensor(self, tensor):
        return(tensor.numpy().tolist())

    def serialize(self):
        serialized_data = {
            'buffer_size': self.buffer_size,
            'strategy': self.strategy,
            'center_strategy': self.center_strategy,
            'replay': {int(key): [self.serialize_tensor(tensor) for tensor in tensor_list] for key, tensor_list in self.replay.items()},
            'average': {int(key): self.serialize_tensor(tensor) for key, tensor in self.average.items()},
            'size': {int(key): value for key, value in self.size.items()},
        }
        return serialized_data
    
   
    def save_to_json(self, filename):
        data = self.serialize()
        with open(filename, 'w') as file:
            json.dump(data, file)
            

    @staticmethod
    def deserialize_tensor(data):
        if isinstance(data, list):
           return torch.tensor(data)
        return data 

    def deserialize(self, data):
        self.buffer_size = data['buffer_size']
        self.strategy = data['strategy']
        self.center_strategy = data['center_strategy']
        self.replay = {
            int(key): [self.deserialize_tensor(value) for value in tensor_list]
            for key, tensor_list in data['replay'].items()
        }
        self.average = {
            int(key): self.deserialize_tensor(value)
            for key, value in data['average'].items()
        }
           
        self.size = {int(key): value for key, value in data['size'].items()}
        
               
        return self

    @staticmethod
    def load_from_json(filename):
        with open(filename, 'r') as file:
            data = json.load(file)
        replay_buffer = ReplayBuffer()
        
        return replay_buffer.deserialize(data)

  

    def get_replay_total_size(self,key):
        size = 0
        for key,value in self.replay:
            size +=  len(self.replay[key])
        return size
    
    def random_sample (self, replay_dict):
        for key, value in replay_dict.items(): 
            data = torch.stack(value) 
            if self.center_strategy == 'mean':
                center = torch.mean(data, dim=0)
            elif self.center_strategy == 'quantile':   
                center = torch.quantile(data, 0.5,dim=0)
            elif self.center_strategy == 'median':   
                center = torch.median(data,dim=0).values
            self.average[key] = center # offline sample average
            self.size[key] = len(replay_dict[key])
  
            if len(value) < self.buffer_size:
                idx = len(value) 
            else:
                idx = self.buffer_size
            positions=(random.sample(range(len(value)), idx))
            new_list = [value[index] for index in positions]
            for  value1 in new_list:
                 if key in self.replay:
                    self.replay[key].append(value1)
                 else: 
                   self.replay[key] = [value1]
                   
    # if buffer is complete selects 1 replay element to be replaced by 1 element of new sample
    def random_update (self, replay_dict,key):
        print('random_update')
        value = replay_dict[key]
        data = torch.stack(value) 
          
        if self.center_strategy == 'mean':
            data_center = torch.mean(data, dim=0)
        elif self.center_strategy == 'quantile':   
            data_center = torch.quantile(data, 0.5,dim=0)
        elif self.center_strategy == 'median':   
            data_center = torch.median(data,dim=0).values   
        nr_elements = len(self.replay[key])
        size_total = self.size[key] + len(replay_dict[key])
        slots = self.buffer_size - nr_elements
        update = 0
        print('SLOT  ', slots)
        if slots == 0:
            if random.random() > 0.90:
               pos_to_be_replaced = random.sample(range(self.buffer_size), 1)[0]
               pos_to_replace = random.sample(range(len(replay_dict[key])), 1)[0]
               print(' pos_to_be_replaced ',  pos_to_be_replaced)
               print(' pos_to_replace ',  pos_to_replace)
               
               self.replay[key][pos_to_be_replaced] = replay_dict[key][pos_to_replace]
               update += 1/self.buffer_size
        else:
            if slots >  len(replay_dict[key]):
                slots = len(replay_dict[key]) 
            positions=(random.sample(range(len(replay_dict[key])), slots))
            print('positions ', positions)
            new_list = [replay_dict[key][index] for index in positions]
            print('new list ', new_list)
            for  value in new_list:
                 if key in self.replay:
                    self.replay[key].append(value)
                 else: 
                    self.replay[key] = [value] 
                 update += 1/self.buffer_size
                 
        # update sample average and number of elements 
        self.average[key] = ((self.size[key]/size_total) * self.average[key]) + ((len(replay_dict[key])/size_total) * data_center)
        print('average[key]')     
        self.size[key] = size_total  
        print('size[key]')
        return (update)           
   
    def exemplar_sample (self, replay_dict):
        for key, value in replay_dict.items(): 
            data = torch.stack(value) 
            if self.center_strategy == 'mean':
                center = torch.mean(data, dim=0)
            elif self.center_strategy == 'quantile':   
                center = torch.quantile(data, 0.5,dim=0)
            elif self.center_strategy == 'median':   
                center = torch.median(data,dim=0).values

            distances = np.linalg.norm(data - center, axis=(1, 2))
            self.average[key] = center # offline sample average
            self.size[key] = len(replay_dict[key]) 
            sorted_indices = np.argsort(distances)
            if len(sorted_indices) < self.buffer_size:
                 idx = len(sorted_indices)
            else:
                 idx = self.buffer_size
            positions = sorted_indices[:idx]
            new_list = [value[index] for index in positions]
            for  value1 in new_list:
                 if key in self.replay:
                    self.replay[key].append(value1)
                 else: 
                   self.replay[key] = [value1]     
  
    def exemplar_update (self, replay_dict, key):
        
        value = replay_dict[key]
        data = torch.stack(value) 
        if self.center_strategy == 'mean':
            data_center = torch.mean(data, dim=0)
        elif self.center_strategy == 'quantile':   
            data_center = torch.quantile(data, 0.5,dim=0)
        elif self.center_strategy == 'median':   
            data_center = torch.median(data,dim=0).values   
        size_total = self.size[key] + len(replay_dict[key])
        size_replay = len(self.replay[key])
        
       
        # update sample average and number of elements 
        self.average[key] = ((self.size[key]/size_total) * self.average[key]) + ((len(replay_dict[key])/size_total) * data_center)
        self.size[key] = size_total
        
        slots = self.buffer_size - size_replay
        center = self.average[key] 
        distances_new = np.linalg.norm(data - center, axis=(1, 2))
        sorted_indices_new = np.argsort(distances_new)
        update = 0
        if slots == 0:
            data_replay = torch.stack(self.replay[key]) 
            distances_replay = np.linalg.norm(data_replay - center, axis=(1, 2))
            max_replay = max(distances_replay)
            sorted_indices_replay = np.argsort(distances_replay)
            descending = sorted_indices_replay[::-1]
            pos_new = [i for i,v in enumerate(distances_new) if v < max_replay]
            for idx, pos in enumerate(pos_new):
                if idx >= size_replay:
                    break
                else:
                    self.replay[key][descending[idx]] = data[pos]
                    update += 1/self.buffer_size
        else:
            positions = sorted_indices_new[:slots]
            new_list = [replay_dict[key][index] for index in positions]
            for  value1 in new_list:
                 self.replay[key].append(value1)
                 update += 1/self.buffer_size           

        return update
                
    def herding_sample (self, replay_dict):
        for key, value in replay_dict.items(): 
             selected_indices = []
             data = torch.stack(value) 
             if self.center_strategy == 'mean':
                center = torch.mean(data, dim=0)
             elif self.center_strategy == 'quantile':   
                center = torch.quantile(data, 0.5,dim=0)
             elif self.center_strategy == 'median':   
                center = torch.median(data,dim=0).values
             current_center = center * 0
             self.average[key] = center
             self.size[key] = len(replay_dict[key])
             for i in range(len(data)):
                 # Compute distances with real center
                 candidate_centers = current_center * i / (i + 1) + data / (
                     i + 1  )
                 distances = np.linalg.norm(candidate_centers - center, axis=(1, 2))
                 distances[selected_indices] = np.inf

                 # Select best candidate
                 new_index = distances.argmin().tolist()
                 selected_indices.append(new_index)
                 current_center = candidate_centers[new_index]
             
             if len(selected_indices) < self.buffer_size:
                 idx = len(selected_indices)
             else:
                 idx = self.buffer_size
             positions = selected_indices[:idx]
             new_list = [value[index] for index in positions]
             for  value1 in new_list:
                  if key in self.replay:
                     self.replay[key].append(value1)
                  else: 
                    self.replay[key] = [value1]    

    # implemented according Incremental Learning In Online Scenario
    # https://arxiv.org/abs/2003.13191

    def herding_update (self, replay_dict, key):
        
        data = replay_dict[key]

        value = replay_dict[key]
        data = torch.stack(value) 
        if self.center_strategy == 'mean':
            data_center = torch.mean(data, dim=0)
        elif self.center_strategy == 'quantile':   
            data_center = torch.quantile(data, 0.5,dim=0)
        elif self.center_strategy == 'median':   
            data_center = torch.median(data,dim=0).values   
        size_total = self.size[key] + len(replay_dict[key])
        size_replay = len(self.replay[key])
        slots = self.buffer_size - size_replay
        update = 0
        if slots == 0:
            for value in data:
                M_old = (self.size[key]/(self.size[key]+1)) * self.average[key] 
                M_factor = (1/(self.size[key]+1))
                M = M_old + (value * M_factor)
                d = []
                for exemplar in self.replay[key]:
                    d.append(np.linalg.norm(exemplar - M))
                sorted_indices = np.argsort(d)
                I_min = sorted_indices[0]
                d_min = d[I_min]
                d_q = np.linalg.norm(value - M)
                if d_q <= d_min:
                   self.replay[key][I_min] = value
                   
                   update += 1/self.buffer_size
        else:    
           current_center = data_center * 0 
           selected_indices = []
           for i in range(len(data)):
               # Compute distances with real center
               candidate_centers = current_center * i / (i + 1) + data / (
                   i + 1
               )
               distances = np.linalg.norm(candidate_centers - data_center, axis=(1, 2))
               distances[selected_indices] = np.inf

               # Select best candidate
               new_index = distances.argmin().tolist()
               selected_indices.append(new_index)
               current_center = candidate_centers[new_index]
           
           positions = selected_indices[:slots]
           new_list = [value[index] for index in positions]
           for  value1 in new_list:
                self.replay[key].append(value1)
                update += 1/self.buffer_size
        
        # update sample average and number of elements 
        self.average[key] = ((self.size[key]/size_total) * self.average[key]) + ((len(replay_dict[key])/size_total) * data_center)
        self.size[key] = size_total
  
        return update
  
                          
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def sample_trajectory(self, batch_size):
        initial_index = random.randint(0, len(self.buffer) - batch_size)
        return self.buffer[initial_index: initial_index + batch_size]
    
    # c = class id
    # sample = list with new class c observations 
    def update_replay(self, sample, c):
        
        if c in self.replay.keys(): # if class is already in replay memory : update
           replay_dict = create_dictionary(sample)
          # print('updating existing class')
           if self.strategy=='random':
              fraction_updating = self.random_update(replay_dict, c)
           elif self.strategy == 'exemplar':
                fraction_updating = self.exemplar_update(replay_dict, c)  
           elif self.strategy == 'herding':
                fraction_updating = self.herding_update(replay_dict, c)  
           else:
                print('strategy not valid')
                sys.exit()
        else:
           fraction_updating = 1
           replay_dict = create_dictionary(sample)
           #print('updating new class')
           if self.strategy=='random':
               self.random_sample(replay_dict)
           elif self.strategy == 'exemplar':
                self.exemplar_sample(replay_dict)  
           elif self.strategy == 'herding':
                self.herding_sample(replay_dict)  
           else:
                print('strategy not valid')
                sys.exit()
        return fraction_updating    
  
    
