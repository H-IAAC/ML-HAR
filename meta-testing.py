
import logging
import os
import sys
import copy
import subprocess
import numpy as np
import pandas as pd
import random
import utils.stats as st
from datetime import datetime

import torch 

from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix

import configs.classification.class_parser_eval_har as class_parser_eval_har
import model.modelfactory as mf
import utils
from experiment.experiment import experiment
from utils.utils import  prepare_json_dataset, prepare_json_stats, average_stats
from datasets.har import get_dataloaders
from model.meta_learner import MetaLearingClassification
import json


logger = logging.getLogger('experiment')


def load_model(args, config):
    
    maml = MetaLearingClassification(args, config)
    
    if args['model_path'] is not None and not args['scratch']:
  
        model = torch.load(args['model_path'],
                         map_location="cpu")
        
        
        for paramA, paramB in zip(maml.net.vars, model.vars):
            paramA.data.copy_(paramB.data.clone())
  
    if (args['reset_weights']):
        maml.net.reset_vars()
   
    return maml

def eval_iterator(iterator, device, maml, args):

    confusion_pred = []
    confusion_act = []
    
    for img, target in iterator:
        with torch.no_grad():
            img = img.to(device)
            target = target.to(device)
            logits_q = maml(img)
    
            pred_q = (logits_q).argmax(dim=1)

            confusion_pred += pred_q.tolist()
            confusion_act += target.tolist()

    labels = sorted(np.unique([confusion_pred, confusion_act]))
    labels_iteration = np.array(list(map(str, labels))) 
    confusion_mat = pd.DataFrame(confusion_matrix(y_true=confusion_act, y_pred=confusion_pred,labels=labels),index = labels_iteration,columns=labels_iteration)  
    return confusion_mat

def train_iterator(iterator, device, maml, opt):

  for X, Y in iterator: 
    X = X.to(device)
    Y = Y.to(device)
    pred = maml(X)
    opt.zero_grad()
    loss = F.cross_entropy(pred, Y)
    loss.backward()
    opt.step()


def generate_stats(my_experiment, base_results_train, base_results_test,stats_id, new_classes, args, label):     

    for data in stats_id:
        if 'Forgetting' not in data and 'Backward' not in data:
            group_data = stats_id.get(data)
            for stats_group in group_data:
                
                data_stats = base_results_train if "Train" in stats_group else base_results_test
                if "classes" not in data:
                    my_experiment.add_result(stats_group+label, average_stats(data_stats))

    return my_experiment


def create_schedule(nr_classes,classes_per_iteration):     

    schedule = []

    for i in range(classes_per_iteration,nr_classes+1,classes_per_iteration): 
        schedule.append(i)
    if schedule[-1] <  nr_classes:
       schedule[-1] = schedule[-1] + (nr_classes - schedule[-1])
     
    return schedule


def main():
    
    python_command =  [sys.executable.split('/')[-1]]
    p = class_parser_eval_har.Parser()
    rank = p.parse_known_args()[0].rank
    all_args = vars(p.parse_known_args()[0])
    print("All args = ", all_args)

    args = utils.get_run(vars(p.parse_known_args()[0]), rank)
    offset_min = 1
    offset_max = 300
    random_offset = random.randint(offset_min, offset_max)
    
    if args['new_seed']:
       args['seed'] = int(datetime.now().timestamp() + 60 * random_offset)
    utils.set_seed(args['seed'])
    
    # open stats metadata for stats computation
    with open(os.getcwd() + args['json_config'], 'r') as f:
        stats_id = json.load(f)
   
    print('path ', args['path'])
    
    args['model_path'] = args['path'] + 'learner.model'
    
    with open(args['path'] +'metadata.json', 'r') as f:
        obj = json.load(f)


    # getting off-line parametrization
    args['labels'] =  obj.get('params').get('labels') 
    args['class_base'] = obj.get('params').get('classes_trajectory') 
    if obj.get('params').get('random'):         
        args['class_base'] += obj.get('params').get('classes_random') 
        
    args['class_new'] = np.setdiff1d(args['labels'],args['class_base']).tolist()
    args['num_classes_dataset'] = obj.get('params').get('number_classes_dataset') 
    args['subject_offline_train'] = obj.get('params').get('subject_offline_train')
    args['subject'] =  obj.get('params').get('subject')
    args['dataset'] = obj.get('params').get('dataset')
    args['scenario'] = obj.get('params').get('scenario')
    args['dataset_path']  = obj.get('params').get('dataset_path')
    args['is_standardized'] = obj.get('params').get('is_standardized')
    args['number_classes_dataset'] =  obj.get('params').get('number_classes_dataset')
    args['channels'] = obj.get('params').get('channels')
    args['data_size'] = obj.get('params').get('data_size')
    args['layers'] = obj.get('params').get('layers')
    args['kernel'] = obj.get('params').get('kernel')
    args['stride'] = obj.get('params').get('stride')
    args['out_linear']  = obj.get('params').get('out_linear')
    
    if 'baseline' in args['path']:
        args['meta_lr'] = obj.get('params').get('lr') 
    else:    
        args['meta_lr'] = obj.get('params').get('meta_lr')
    args['update_lr'] = obj.get('params').get('update_lr')
    args['update_step'] = obj.get('params').get('update_step')
    args['query'] =  obj.get('params').get('query')
    args['random'] =  obj.get('params').get('random')
    args['runs_offline'] = obj.get('params').get('runs')
    args['reset'] =  obj.get('params').get('reset')
    
    
    if args['scenario'] == 'nic':
       print('meta-learning scenario  not valid - must be nc (new classes)') 
       sys.exit()
   
    args['main_folder'] =  obj.get('params').get('main_folder')
    
    args['augmentation_ref'] = obj.get('params').get('augmentation_ref')
    
    my_experiment = experiment(args['name'], args, args['folder_id'] + args['dataset'] + "/" + args['augmentation_ref'] + "/"  , commit_changes=False, rank=args['runs_offline'], seed=1)
       
    print('my_experiment.path ',my_experiment.path)
    
    writer = SummaryWriter(my_experiment.path + "tensorboard")
    
    classes_new = [int(element) for element in args['class_new']]
    classes_base = [int(element) for element in args['class_base']]
    
    print('classes new: ', classes_new)
    print('classes base: ', classes_base)
    
    
    args['subject_online'] =  args['subject']  
           
    # prints for validation
    print('\n args[scenario]',args['scenario'])
    print('\n args[subject_offline_train]',args['subject_offline_train'])
    print('\n args[subject_online]',args['subject_online'])
    print('\n no_of_classes_schedule',  args['classes_schedule'])
    
    args['dataset_path'] = None
    
    # get train and test data
    
    data = get_dataloaders(args['dataset'],
                           args['dataset_path'],
                           is_train=True,
                           batch_size=1,
                           is_standardized=args['is_standardized'],
                           dataloader=False)

    data_test = get_dataloaders(args['dataset'],
                                args['dataset_path'],
                                is_train=False,
                                batch_size=1,
                                is_standardized=args['is_standardized'],
                                dataloader=False)
    
    
    my_experiment.results["Class info"] = prepare_json_dataset(data)
    
    
    # remove from train and test set classes seen in offline training
    data_train = utils.remove_classes_ucihar(data, classes_new) 
    print('\ndata_train nc', np.unique((data_train.Y).numpy()))
        
    data_test = utils.remove_classes_ucihar(data_test, classes_new ) 
    print('\ndata_test nc', np.unique((data_test.Y).numpy()))
        
    # get train and test data to evaluate results on base classes
    
    data_train_base = get_dataloaders(args['dataset'],
                                  args['dataset_path'],
                                  is_train=True,
                                  batch_size=1,
                                  is_standardized=args['is_standardized'],
                                  dataloader=False)

    data_test_base = get_dataloaders(args['dataset'],
                                  args['dataset_path'],
                                  is_train=False,
                                  batch_size=1,
                                  is_standardized=args['is_standardized'],
                                  dataloader=False)
    
       
    # remove new classes from train and test samples for evaluating performance on base classes
    
    dataset = utils.remove_classes_ucihar(data_train_base, classes_base)
    
    iterator_train_base = torch.utils.data.DataLoader(dataset, batch_size=32,
                                     shuffle=False, num_workers=1)

    
    print('\niterator_train base', np.unique(iterator_train_base.dataset.Y))
    
    dataset = utils.remove_classes_ucihar(data_test_base, classes_base)
    iterator_test_base = torch.utils.data.DataLoader(dataset, batch_size=32,
                                                shuffle=False, num_workers=1)
     
    
    print('\niterator_test base', np.unique(iterator_test_base.dataset.Y))
               
    results_train = []
    results_test = []
    results_train_json = []
    results_test_json = []
    base_results_train = []
    base_results_test = []
    base_results_train_json = []
    base_results_test_json = []


    gpu_to_use = rank % args["gpus"]
    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(gpu_to_use))
        logger.info("Using gpu : %s", 'cuda:' + str(gpu_to_use))
    else:
        device = torch.device('cpu')

   
    config = mf.ModelFactory.get_model("na", dataset='har_1layer', 
                                        output_dimension=args['number_classes_dataset'], 
                                        channels=args['channels'],
                                        data_size = args['data_size'],
                                        cnn_layers = args['layers'],
                                        kernel = args['kernel'],
                                        stride = args['stride'],
                                        out_linear = args['out_linear'])


    
    maml = load_model(args, config)
    
   
    weights = copy.deepcopy(maml.net.vars)

    maml = maml.to(device)

    
    args['schedule'] = create_schedule(len(args['class_new']), args['classes_schedule'])
    print('schedule: ', args['schedule'])     

    for total_classes in args['schedule']:
        # search for best learning rate
        lr_sweep_results = []
        lr_sweep_range = [0.03, 0.01, 0.003,0.001, 0.0003, 0.0001, 0.00003, 0.00001]
        lr_all = []
    
        print('total_classes', total_classes)    
        
        #search for 'best' learning rate
        
        for lr_search_runs in range(0, 5):
            
            
            classes_to_keep = np.random.choice(classes_new, total_classes, replace=False).tolist()
            
            print('classes to keep lr: ', classes_to_keep)
            
            dataset = utils.remove_classes_ucihar(data_train, classes_to_keep)
        
            iterator_sorted = torch.utils.data.DataLoader(
                utils.iterator_sorter_omni(dataset, False, classes=args['schedule']),
                batch_size=1,
                shuffle=args['iid'], num_workers=2)
            
            
            print('\niterator_sorted lr', np.unique(iterator_sorted.dataset.Y))

            iterator_train_lr = torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'],
                                                         shuffle=False, num_workers=1)   
            
            print('\niterator_train lr', np.unique(iterator_train_lr.dataset.Y))

            max_acc = -1000

            for lr in lr_sweep_range:

        
                if (args['reset_weights']):
                   maml.net.reset_vars()
                else:
                   maml.net.update_weights(weights)
                
                opt = torch.optim.Adam(maml.net.get_adaptation_parameters(), lr=lr)
        
                train_iterator(iterator_sorted, device, maml.net, opt)
                
                confusion_mat = eval_iterator(iterator_train_lr, device, maml.net, args)
                stats= st.Stats(confusion_mat, lr) 
                train_accuracy = stats.get_f1(weighted=True, macro=True)
                if (train_accuracy > max_acc):
                    max_acc = train_accuracy
                    max_lr = lr


            lr_all.append(max_lr)
            results_mem_size = (max_acc, max_lr)
            

            lr_sweep_results.append([total_classes, results_mem_size])
        
            my_experiment.results["LR Search Results"] = lr_sweep_results
            my_experiment.store_json()
            logger.debug("LR RESULTS = %s", str(lr_sweep_results))
        
        from scipy import stats as st1
        best_lr = float(st1.mode(lr_all)[0][0])
        args['best_lr'] = best_lr
    
        logger.info("BEST LR %s= ", str(best_lr))    
        
               
        lr = best_lr
        
       
        for current_run in range(0, args['runs']):
                
            classes_to_keep = np.random.choice(classes_new, total_classes, replace=False).tolist()
 
            label_classes_to_keep =  [str(element) for element in classes_to_keep]
            
            print('classes to keep training: ', classes_to_keep)
            print('classes base: ', args['class_base'])
            
            dataset = utils.remove_classes_ucihar(data_train, classes_to_keep)
            
            iterator_sorted = torch.utils.data.DataLoader(
               utils.iterator_sorter_omni(dataset, False, classes=args['schedule']),
                batch_size=1,
                shuffle=args['iid'], num_workers=2)
            
            # remove classes not selected for training from train and test samples
            
   
            dataset = utils.remove_classes_ucihar(data_train, classes_to_keep)
            
            iterator_train = torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'],
                                             shuffle=False, num_workers=1)
   
            dataset = utils.remove_classes_ucihar(data_test, classes_to_keep)
            iterator_test = torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'],
                                                        shuffle=False, num_workers=1)
                        
            
             # reset to appropriate model
              
            if (args['reset_weights']):
               maml.net.reset_vars()
            else:
                maml.net.update_weights(weights)
                          
           
            opt = torch.optim.Adam(maml.net.get_adaptation_parameters(), lr=lr)
    
          
            train_iterator(iterator_sorted, device, maml.net, opt)
            
            stats  = utils.log_accuracy_har_v2(maml.net, my_experiment, iterator_train, device, writer, current_run, label_classes_to_keep, 'Train', args)

            results_train.append(stats)

            results_train_json.append(prepare_json_stats(stats))
 
            stats = utils.log_accuracy_har_v2(maml.net, my_experiment, iterator_test, device, writer, current_run, label_classes_to_keep, 'Test', args)

            results_test.append(stats)

            results_test_json.append(prepare_json_stats(stats))
            


            
            # stats base dataset  
    
            stats  = utils.log_accuracy_har_v2(maml.net, my_experiment, iterator_train_base, device, writer, current_run, classes_base, 'Train', args)

            base_results_train.append(stats)

            base_results_train_json.append(prepare_json_stats(stats))

            stats = utils.log_accuracy_har_v2(maml.net, my_experiment, iterator_test_base, device, writer, current_run, classes_base, 'Test', args)

            base_results_test.append(stats)

            base_results_test_json.append(prepare_json_stats(stats))
    
       
        my_experiment.add_result("Train", results_train_json)
        my_experiment.add_result("Test", results_test_json)
        
        my_experiment.add_result("Train base", base_results_train_json)
        my_experiment.add_result("Test base", base_results_test_json)

        my_experiment = generate_stats(my_experiment, base_results_train, base_results_test, stats_id, label_classes_to_keep, args, ' base ' + str(total_classes))
        my_experiment = generate_stats(my_experiment, results_train, results_test, stats_id, label_classes_to_keep, args, ' ' + str(total_classes))

        my_experiment.store_json()


    if args['plot']:
        print('PLOT') 
        arguments_list = [args['plot_file'], "--path", os.path.abspath(my_experiment.path)+'/']
        print('ARGUMENT LIST', arguments_list)
        try:
            result = subprocess.run(python_command + arguments_list, check=True, capture_output=False, text=True)
            print("Command output:", result.stdout)
        except subprocess.CalledProcessError as e:
            print("Error occurred:", e)
            print("Command output (if available):", e.stdout)
            print("Command error (if available):", e.stderr)
        else:
            print("plotting execution based successfully.")
              
if __name__ == '__main__':
    main()
