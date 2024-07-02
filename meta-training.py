import logging
import copy
import subprocess
import os

import numpy as np
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datetime import datetime

import configs.classification.class_parser_har as class_parser_har
import model.modelfactory as mf
import utils.utils as utils
from utils.utils import sample_subject,sample_metatest_data, prepare_json_stats, prepare_json_dataset
from experiment.experiment import experiment
from model.meta_learner import MetaLearingClassification
from datasets.utils import concat_samples
from datasets.har import  get_dataloaders


def main():
    
    python_command =  [sys.executable.split('/')[-1]]
    p = class_parser_har.Parser()
    rank = p.parse_known_args()[0].rank
    all_args = vars(p.parse_known_args()[0])
    print("All args = ", all_args)
        
    args = utils.get_run(vars(p.parse_known_args()[0]), rank)
    
    if args['model'] == 'oml' and not args['random']:
       print('For oml model, random must be True')
       sys.exit()
    if args['model'] == 'maml' and args['random']:
       print('For maml model, random must be False')
       sys.exit() 
        
    # prepare augmentation 
    
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

    iterator_train_complete = get_dataloaders(args['dataset'],
                                  args['dataset_path'],
                                  is_train=True,
                                  batch_size=args['batch_size'],
                                  is_standardized=args['is_standardized'])
    
    for run in range(args['runs']): 
        
        print('\n run: ', run)
    
        if args['new_seed']:
           args['seed'] = int(datetime.now().timestamp())
        
        print('\n seed ',args['seed'])
    
        utils.set_seed(args['seed'])
       
        # PREPARES LOGGERS
       
        my_experiment = experiment('', args, "../" + args['main_folder'] +"/" + args['name'] +  "/" +  args['model'] +  "/" + args['scenario'] +  "/" + args['dataset'] + "/" + dsc  , commit_changes=False, rank=args['steps'], seed=1)
        
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
        
        print(' random_labels    ', random_labels)
        
        if args['random']:
            print('random')
            args['classes_trajectory'] = random_labels[0:round(number_classes/2)]
            args['classes_random'] =   random_labels[round(number_classes/2):]
        else:
            print('not random')
            args['classes_trajectory'] = random_labels[0:round(number_classes)]
            args['classes_random'] = ''
    
        # print for validation
        print('\nargs[classes_trajectory] ', args['classes_trajectory'] )
        print('\nargs[classes_random] ', args['classes_random'] )
        print('\nargs[labels] ',     args['labels'] )
              
        if args['random']:
            args['label_training']  = args['classes_random'] + args['classes_trajectory']
        else:
            args['label_training'] = args['classes_trajectory'] 
        
          
        print('\nargs[label_training] ', args['label_training'] )
         
        print('\nargs[classes_trajectory] after', args['classes_trajectory'] )
       
        classes_trajectory = np.array(list(map(int, args['classes_trajectory'])))
       
        classes_random = np.array(list(map(int, args['classes_random'])))
    
        print('classes_trajectory ', classes_trajectory)
        print('classes_random ', classes_random)
    
        # setting subject to sample data
        
        args['subject'] = train_loader.get_subject_id()
        
               
        args['subjects_candidate'] = sample_subject(train_loader,
                                    target = args['label_training'],
                                    root=args['dataset_path'], 
                                    group='train'
                                   )
        
        args['subject_offline_train'] = args['subjects_candidate']
       
        # print for validation
        print('\nargs[subject] ', args['subject'] )
        print('\nargs[subject_offline_train] ', args['subject_offline_train'] )
      
        # sample sujects 
        data_train = sample_metatest_data(train_loader,
                                          target=args['subject_offline_train'],
                                          root=args['dataset_path'], 
                                          group='train', 
                                          task='subject')
        
        
        
        
        # selects trajectory classes
        
        dataset_trajectory = utils.remove_classes_ucihar(data_train, classes_trajectory)
        print('\ndataset trajectory classes', np.unique((dataset_trajectory.Y).numpy()))
        
        # selects random classes
        if args['random']:
            dataset_random = utils.remove_classes_ucihar(data_train, classes_random) 
            print('\ndataset random classes',np.unique((dataset_random.Y).numpy()))
        else:
            dataset_random= ''
       
            
        # PREPARES DATA EVALUATION
         
        # train - keeps classes in trajectory and random for the subjects uses to training
        
        dataset_train_eval = copy.deepcopy(dataset_trajectory)
        if args['random']:
            dataset_train_eval  = concat_samples(dataset_train_eval ,dataset_random)
         
        # creates iterator                                     
        iterator_train = DataLoader(dataset_train_eval,
                              batch_size=args['batch_size'],
                              shuffle=True)
        print('\ndataset_train eval', np.unique((dataset_train_eval.Y).numpy()))
       # test - keeps classes in trajectory and random sets in test set (different users)
        print('\niterator_test', np.unique((iterator_train.dataset.Y).numpy()))
        test_loader = get_dataloaders(args['dataset'],
                                      args['dataset_path'],
                                      is_train=False,
                                      batch_size=1,
                                      is_standardized=args['is_standardized'],
                                      dataloader=False)
        
        # sample only classes used to learning
        dataset_test = utils.remove_classes_ucihar(test_loader, np.concatenate((classes_random,classes_trajectory)))
        print('\ndataset_test', np.unique((dataset_test.Y).numpy()))
       
        #creates iterator
        iterator_test = DataLoader(dataset_test,
                              batch_size=args['batch_size'],
                              shuffle=True)
       
        # selects data_train and data_test to evaluate - entire dataset
        print('\niterator_test', np.unique((iterator_test.dataset.Y).numpy()))
        iterator_test_complete = get_dataloaders(args['dataset'],
                                      args['dataset_path'],
                                      is_train=False,
                                      batch_size=args['batch_size'],
                                      is_standardized=args['is_standardized'])
    
        
        # PREPARES MODEL
        
        
        config = mf.ModelFactory.get_model("na", dataset=args['network_id'],
                                           output_dimension=args['number_classes_dataset'], 
                                           channels=args['channels'],
                                           data_size = args['data_size'],
                                           cnn_layers = args['layers'],
                                           kernel = args['kernel'],
                                           stride = args['stride'],
                                           out_linear = args['out_linear'])
    
        
        print('config ', config)    
        my_experiment.results["Class info"] = prepare_json_dataset(data_train)
        
        gpu_to_use = rank % args["gpus"]
        if torch.cuda.is_available():
            device = torch.device('cuda:' + str(gpu_to_use))
            logger.info("Using gpu : %s", 'cuda:' + str(gpu_to_use))
        else:
            device = torch.device('cpu')
    
        maml = MetaLearingClassification(args, config).to(device)

        avg_acc = 0
        acc_loss = []
          
        #print('maml.net ', maml.net)
        
        for step in range(args['steps']): 
            
            print('step ', step)

            t = maml.select_classes2train(classes_trajectory , args['tasks'])
            
            #print('tasks ', t)
                 
            x_spt, y_spt, x_qry, y_qry = maml.select_samples2train_new(dataset_trajectory, t, dataset_random, 
                                                                       classes_random,
                                                                       num_support=args['update_step'], num_query=args['query'], 
                                                                       random=args['random'], reset = args['reset']) 
            if torch.cuda.is_available():
                x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
                
            accs, loss = maml(x_spt, y_spt, x_qry, y_qry)
            
            avg_acc += accs[-1]
      
            # Evaluation during training for sanity checksi
            if step % 100 == 0:
               #writer.add_scalar('F:/abordagens/mrcl/experiment', accs[-1], step)
               #logger.info('step: %d \t training acc %s', step, str(accs))
               result_loss = [tensor.item() for tensor in loss]
               spt = [tensor.item() for tensor in y_spt] 
               results = {"step": step, "acc": accs.tolist(), "loss": result_loss, "y_spt": spt, "y_qry": y_qry.tolist()}
               acc_loss.append(results)
        

     
        my_experiment.add_result("Learning stats", acc_loss)
      
       # train and test evaluation according to training data
     
        stats = utils.log_accuracy_har_v2(maml.net, my_experiment, iterator_train, device, writer, step, args['labels'], 'Train', args)
    
        my_experiment.add_result("Train", prepare_json_stats(stats))
        
        stats = utils.log_accuracy_har_v2(maml.net, my_experiment, iterator_test, device, writer, step,args['labels'], 'Test', args)
    
        my_experiment.add_result("Test", prepare_json_stats(stats))
        
        # train and test evaluation entire dataset
      
        stats = utils.log_accuracy_har_v2(maml.net, my_experiment, iterator_train_complete, device, writer, step, args['labels'], 'Train', args)
    
        my_experiment.add_result("Train Complete", prepare_json_stats(stats))
         
        stats = utils.log_accuracy_har_v2(maml.net, my_experiment, iterator_test_complete, device, writer, step,args['labels'], 'Test', args)
    
        my_experiment.add_result("Test Complete", prepare_json_stats(stats))
     
        torch.save(maml.net, my_experiment.path + "learner.model")
        
        my_experiment.store_json()
    
        # plotting graphics with stats
        
          
        if args['plot']:
            arguments_list = [args['plot_file'], "--path", os.path.abspath(my_experiment.path)+'/']
            print("Running offline plotting.")
            try:
                result = subprocess.run(python_command + arguments_list, check=True, capture_output=False, text=True)
                print("Command output:", result.stdout)
            except subprocess.CalledProcessError as e:
                print("Error occurred:", e)
                print("Command output (if available):", e.stdout)
                print("Command error (if available):", e.stderr)
            else:
                print("plotting execution completed successfully.")
        if torch.cuda.is_available():        
           torch.cuda.empty_cache()   
        
 
     
if __name__ == '__main__':
    main()
 
