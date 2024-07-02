
import logging
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import utils.stats as st
from datetime import datetime
import torch

from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix

import configs.classification.class_parser_baseline as class_parser_baseline
import model.learner as Learner
import model.modelfactory as mf
import utils
from experiment.experiment import experiment
from utils.utils import  prepare_json_stats, prepare_json_stats_baseline, prepare_json_dataset, sample_metatest_data
from datasets.har import get_dataloaders

logger = logging.getLogger('experiment')


def set_model(args, config):
    
    net = Learner.Learner(config)
    
    return net


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

        
def train_iterator(iterator, device, maml, opt, args):
    
  counter = 0
  sum_loss = 0
  
  confusion_pred = []
  confusion_act = []
   
  for X, Y in iterator: 
    X = X.to(device)
    Y = Y.to(device)
    pred = maml(X)
    opt.zero_grad()
    loss = F.cross_entropy(pred, Y)
    
    pred_q = (pred).argmax(dim=1)
    confusion_pred += pred_q.tolist()
    confusion_act += Y.tolist()

  
    
    # Add L2 regularization term to the loss
    if args['l2']:
       l2_reg = sum(torch.norm(param) for param in maml.parameters())
       loss += args['l2_lambda'] * l2_reg
   
    sum_loss += loss.item()
    counter += 1
    loss.backward()
    opt.step()

    labels = sorted(np.unique([confusion_pred, confusion_act]))
    labels_iteration = np.array(list(map(str, labels))) 
    confusion_mat = pd.DataFrame(confusion_matrix(y_true=confusion_act, y_pred=confusion_pred,labels=labels),index = labels_iteration,columns=labels_iteration)  

    
  return [sum_loss/counter, confusion_mat]



def main():
    
    python_command =  [sys.executable.split('/')[-1]]
    p = class_parser_baseline.Parser()
    rank = p.parse_known_args()[0].rank
    all_args = vars(p.parse_known_args()[0])
    print("All args = ", all_args)

    args = utils.get_run(vars(p.parse_known_args()[0]), rank)

   # prepares augmentation 
    
    dsc = 'None'
    
    print('AUG ', args['augmentation'])
    
    if args['augmentation'] is not None:
        dsc = ''
        if 'Jitter' in args['augmentation']:
           dsc += 'J'
        if 'Scale' in args['augmentation']:
           dsc += 'S' 
        if 'Perm' in args['augmentation']:
           dsc += 'P' 
        if 'MagW' in args['augmentation']:
               dsc += 'M'     
        if 'TimeW' in args['augmentation']:
            dsc += 'T'
    print('dsc ', dsc)   
    

    train_loader = get_dataloaders(args['dataset'],
                                   args['dataset_path'],
                                   is_train=True,
                                   batch_size=1,
                                   is_standardized=args['is_standardized'],
                                   dataloader=False,
                                   data_augmentation = args['augmentation'])
    
    args['augmentation_ref'] = dsc

    test_loader = get_dataloaders(args['dataset'],
                                   args['dataset_path'],
                                   is_train=False,
                                   batch_size=1,
                                   is_standardized=args['is_standardized'],
                                   dataloader=False)
    
     
    for run in range(args['runs']):    
    
        print('\n run: ', run)
    
        if args['new_seed']:
           args['seed'] = int(datetime.now().timestamp())
        
        print('\n seed ',args['seed'])
    
        utils.set_seed(args['seed'])
        
        # PREPARES LOGGERS
          
        my_experiment = experiment(args['augmentation_ref'] , args, args['folder_id'] + args['name'] + "/" + args['scenario'] + "/"  + args['dataset'] + "/"  , commit_changes=False, rank=args['runs'], seed=1)
       
       
        print(' path ' , my_experiment.path)
      
        writer = SummaryWriter(my_experiment.path + "tensorboard")
        
          
        logger = logging.getLogger('experiment')    
    
        
        if args['dataset_path'] is None:
                args['dataset_path']  = train_loader.get_dataset_path()
                
        # setting class labels 
        args['labels'] =  [str(i) for i in train_loader.get_class_labels()]
       
        # setting trajectory and random classes
        
        number_classes_dataset = train_loader.get_num_classes()
       
        number_classes = round(number_classes_dataset * args['fraction_classes'])
        
        args['number_classes_dataset'] = number_classes_dataset
        
        args['data_size'] = train_loader.get_data_size()
        
        print('data size' , args['data_size'] )
        

        random_positions = np.random.choice(len(args['labels']), number_classes,replace=False)
   
        random_labels = [args['labels'][pos] for pos in random_positions]
        
        args['classes_trajectory'] = random_labels[0:round(number_classes)]
        args['label_training'] = args['classes_trajectory'] 
   
        # print for validationruns
        print('\nargs[classes_trajectory] ', args['classes_trajectory'] )
        print('\nargs[labels] ',     args['labels'] )
          
        print('\nargs[label_training] ', args['label_training'] )
       
        classes_trajectory = np.array(list(map(int, args['classes_trajectory'])))
       
        print('classes_trajectory ', classes_trajectory)
    
        # setting subject to sample data
        
        args['subject'] = train_loader.get_subject_id()
        
        if args['scenario'] == 'nic':
            number_subject = round(len(args['subject']) * args['fraction_subject'])
            random_positions = np.random.choice(len(args['subject']), number_subject,replace=False)
            args['subject_offline_train'] = [args['subject'][pos] for pos in random_positions]
        else:
            args['subject_offline_train'] = args['subject']

       
        print('subject dataset ', args['subject'])
        print('subject subject_offline_train ', args['subject_offline_train'])

            

        # sample sujects 
        data_train = sample_metatest_data(train_loader,
                                          target=args['subject_offline_train'],
                                          root=args['dataset_path'], 
                                          group='train', 
                                          task='subject')
        

        
        # selects trajectory classes
        
        data_train = utils.remove_classes_ucihar(data_train, classes_trajectory)
        print('\ndataset trajectory classes', np.unique((data_train.Y).numpy()))
 
  
        my_experiment.results["Class info"] = prepare_json_dataset(data_train)  
        
        
        dataset_tmp = utils.remove_classes_ucihar(test_loader, classes_trajectory)
        iterator_test = torch.utils.data.DataLoader(dataset_tmp, batch_size=32,
                                                shuffle=False, num_workers=1)
     
    
        print('\niterator_test ', np.unique(iterator_test.dataset.Y))
        
      
        
        
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
    
        print('config', config)
        maml = set_model(args, config)
        
        maml.reset_vars()
       
        maml = maml.to(device)
    
        iterator_sorted = torch.utils.data.DataLoader(
            utils.iterator_sorter_omni(data_train, no_sort=True, random=True, classes=number_classes_dataset),
            batch_size=args['batch_size'],
            shuffle=args['iid'], num_workers=2)
        
     
        if args['l2']:  
           opt = torch.optim.Adam(maml.parameters(), lr=args['lr'], weight_decay=args['l2_lambda'])      
        else:
           opt = torch.optim.Adam(maml.parameters(), lr=args['lr'])
           
            
        if args['decay']:
        
            scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args['decay_factor'])
    
        
        train_results = []

        for run in range(0,args['steps']):
    
            loss, confusion_mat = train_iterator(iterator_sorted, device, maml, opt, args)
            stats= st.Stats(confusion_mat, run) 
            acc = stats.get_f1(weighted=True, macro=True)
           
            print(f"Epoch [{run}/{args['runs']}] - Train Loss: {loss:.4f}, Train Accuracy: {acc:.4f}")
            train_results.append([run, loss,acc])
            
            if args['decay'] and run % args['schedule'] == 0:
                print('\nschedule')
                scheduler.step()
          
        my_experiment.add_result("Epochs", prepare_json_stats_baseline(train_results))
        
        stats  = utils.log_accuracy_har_v2(maml, my_experiment, iterator_sorted, device, writer, args['runs'], args['labels'], 'Train', args)
    
        my_experiment.add_result("Train", prepare_json_stats(stats))
                  
        stats = utils.log_accuracy_har_v2(maml, my_experiment, iterator_test, device, writer, args['runs'], args['labels'], 'Test', args)
     
        my_experiment.add_result("Test", prepare_json_stats(stats))
    
        my_experiment.store_json()
        
        torch.save(maml, my_experiment.path + "learner.model")
        
        arguments_list = [args['plot_file'], "--path", os.path.abspath(my_experiment.path)+'/']
            
        if args['plot']:
           try:
               result = subprocess.run(python_command + arguments_list, check=True, capture_output=True, text=True)
               print("Command output:", result.stdout)
           except subprocess.CalledProcessError as e:
                print("Error occurred:", e)
                print("Command output (if available):", e.stdout)
                print("Command error (if available):", e.stderr)
        else:
                print("plotting execution completed successfully.")
                
if __name__ == '__main__':
    main()