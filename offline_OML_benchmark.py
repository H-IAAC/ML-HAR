import logging
import subprocess
import os
import gc
import csv
import numpy as np
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datetime import datetime

import configs.classification.class_parser_har as class_parser_har
import model.modelfactory as mf
import utils.utils as utils
from utils.utils import sample_subject,sample_metatest_data, prepare_json_stats, ReplayBuffer, prepare_json_dataset
from experiment.experiment import experiment
from model.meta_learner import MetaLearingClassification
from datasets.utils import  standardization_pass_fail_all_modes, check_standardization_adapter_all_modes, to_numpy_3d, standarize_data
from datasets.har_teste import  get_dataloaders
from datasets.augmentation import TimeSeriesAugmenter




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
    

    args['augmentation_ref'] = dsc    
    
    print(args['standardization_mode'])
    print('augmentation ', dsc)
    
 
    train_dataset_raw = get_dataloaders(args['dataset'],
                                   args['dataset_path'],
                                   is_train=True,
                                   batch_size=1,
                                   is_standardized=False,
                                   dataloader=False,                                   
                                   data_augmentation = None)
    
    
    test_dataset_raw = get_dataloaders(args['dataset'],
                                   args['dataset_path'],
                                   is_train=False,
                                   batch_size=1,
                                   is_standardized=False,
                                   dataloader=False,                                   
                                   data_augmentation = None)
    
        
    ''' example augmentation with dataloader
    train_dataset_aug = get_dataloaders(args['dataset'],
                                   args['dataset_path'],
                                   is_train=True,
                                   batch_size=1,
                                   is_standardized=False,
                                   dataloader=False,                                   
                                   data_augmentation = args['augmentation'])
    

    '''
     
    
    
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
                args['dataset_path']  = train_dataset_raw.get_dataset_path()
                
        # setting class labels 
        args['labels'] =  [str(i) for i in train_dataset_raw.get_class_labels()]
       
        # setting trajectory and random classes
        
        number_classes_dataset = train_dataset_raw.get_num_classes()
       
        number_classes = round(number_classes_dataset * args['fraction_classes'])
        
        args['number_classes_dataset'] = number_classes_dataset
        
        
        random_positions = np.random.choice(len(args['labels']), number_classes,replace=False)
   
        random_labels = [args['labels'][pos] for pos in random_positions]
        
        
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
      
      
        classes_trajectory = np.array(list(map(int, args['classes_trajectory'])))
       
        classes_random = np.array(list(map(int, args['classes_random'])))
    
        print('classes_trajectory ', classes_trajectory)
        print('classes_random ', classes_random)
    
        # setting subject to sample data
        
        args['subject'] = train_dataset_raw.get_subject_id()
        
               
        args['subjects_candidate'] = sample_subject(train_dataset_raw,
                                    target = args['label_training'],
                                    root=args['dataset_path'], 
                                    group='train'
                                   )
        
        
        if args['scenario'] == 'nic':
            number_subject = round(len(args['subjects_candidate']) * args['fraction_subject'])
            random_positions = np.random.choice(len(args['subjects_candidate']), number_subject,replace=False)
            args['subject_offline_train'] = [args['subjects_candidate'][pos] for pos in random_positions]
        else:
            args['subject_offline_train'] = args['subjects_candidate']
       
        # print for validation
        print('\nargs[subject] ', args['subject'] )
        print('\nargs[subject_offline_train] ', args['subject_offline_train'] )
      
        # sample sujects raw dataset
        data_train_raw = sample_metatest_data(train_dataset_raw,
                                          target=args['subject_offline_train'],
                                          root=args['dataset_path'], 
                                          group='train', 
                                          task='subject')
        
        print('\ntrain dataset  classes complete ', np.unique((data_train_raw.Y).numpy()))
        
        # sample offline training classes (trajectory + random classes) from raw training dataset
                     
        data_train_raw = utils.remove_classes_ucihar(data_train_raw, np.concatenate([classes_trajectory, classes_random], axis=0))    
        
        data_test_raw = utils.remove_classes_ucihar(test_dataset_raw, np.concatenate([classes_trajectory, classes_random], axis=0))    
       
        print('\ntrain dataset classes offline', np.unique((data_train_raw.Y).numpy()))
        
        print('\ntest dataset classes offline', np.unique((data_test_raw.Y).numpy()))
        
        # standarize training data with training classes (trajectory + random)
        
        X_train_raw_np = to_numpy_3d(data_train_raw.X, channels_last=False).copy()
        X_test_raw_np  = to_numpy_3d(data_test_raw.X,  channels_last=False).copy()

        train_std, test_std = standarize_data(train_dataset=data_train_raw,
                                              test_dataset=data_test_raw,
                                              mode=args['standardization_mode'],
                                              time_cut_ratio=args['time_cut_ratio'],
                                              eps=args['eps'],
                                              apply_to_train=True)
        
        
        
        report = check_standardization_adapter_all_modes(
            X_train_raw=X_train_raw_np,
            X_train_z  =to_numpy_3d(train_std.X, channels_last=False),
            X_test_raw =X_test_raw_np,
            X_test_z   =to_numpy_3d(test_std.X,  channels_last=False),
            mu_train   =np.asarray(train_std.mu),   # keep original shapes (1,1,T)
            sd_train   =np.asarray(train_std.sd),
            mode=args['standardization_mode'],
            time_cut_ratio=args['time_cut_ratio'],
            eps=args['eps'], tol_mean=args['tol_mean'], tol_std=args['tol_std']
        )
        
        ok, msg = standardization_pass_fail_all_modes(report, tol_consistency=1e-4)
        
        print('standardization train and test data datasets by training set mu and sd', ok)
        print('standardization train and test data datasets by training set mu and sd', msg)
       
        mu_train = np.asarray(train_std.mu).tolist()   # shape (1,1,T)
        sd_train = np.asarray(train_std.sd).tolist()   # shape (1,1,T)
  
        mu_test = np.asarray(test_std.mu).tolist()   # shape (1,1,T)
        sd_test = np.asarray(test_std.sd).tolist()   # shape (1,1,T)
        
        my_experiment.add_result("Standardization train - mu", mu_train) 
        my_experiment.add_result("Standardization train - sd", sd_train) 
        my_experiment.add_result("Standardization test  - mu", mu_test) 
        my_experiment.add_result("Standardization test  - sd", sd_test) 
        

        if args['augmentation'] is not None:      
           '''
           # example without append to train_std
           augmenter = TimeSeriesAugmenter(time_axis=2, channel_axis=1)
           X_aug, Y_aug = augmenter.augment_dataset(train_std, args['augmentation'], append_to_dataset=False)

           # example with append to train_std   
           augmenter.augment_dataset(train_std,args['augmentation'], append_to_dataset=True)
           
           '''
           augmenter = TimeSeriesAugmenter(time_axis=2, channel_axis=1)
           augmenter.augment_dataset(train_std, data_augmentation=args['augmentation'], append_to_dataset=True)


        args['data_size'] = train_std.get_data_size()
        
        args['nr_samples'] = train_std.nr_samples
        
        
        # split standardized train sets into trajectory and random 
 
        dataset_trajectory = utils.remove_classes_ucihar(train_std, classes_trajectory)
        print('\ndataset trajectory classes', np.unique((dataset_trajectory.Y).numpy()))
        
        # selects random classes
        if args['random']:
            dataset_random = utils.remove_classes_ucihar(train_std, classes_random) 
            print('\ndataset random classes',np.unique((dataset_random.Y).numpy()))
        else:
            dataset_random= ''
       
             
        # PREPARES DATA EVALUATION
         
        
        # creates iterator standardized training dataset (with trajectory and random classes)                                    
        
        iterator_train = DataLoader(train_std,
                              batch_size=args['batch_size'],
                              shuffle=True)
        print('\ndataset_train eval', np.unique((iterator_train.dataset.Y).numpy()))
        
        
        # creates iterator standardized testing dataset (with trajectory and random classes)    (different users)
               
        iterator_test = DataLoader(test_std,
                              batch_size=args['batch_size'],
                              shuffle=True)
        
        print('\niterator_test', np.unique((iterator_test.dataset.Y).numpy()))
       
      
        # create iterators for train_dataset_std and test_dataset_std to evaluate - entire dataset
        
       
        X_train_dataset_raw_np = to_numpy_3d(train_dataset_raw.X, channels_last=False).copy()
        X_test_dataset_raw_np  = to_numpy_3d(test_dataset_raw.X,  channels_last=False).copy()

        train_complete_std, test_complete_std = standarize_data(train_dataset=train_dataset_raw,
                                              test_dataset=test_dataset_raw,
                                              mode=args['standardization_mode'],
                                              time_cut_ratio=args['time_cut_ratio'],
                                              eps=args['eps'],
                                              mu=mu_train,
                                              sd=sd_train,                                              
                                              apply_to_train=True)
        
        report = check_standardization_adapter_all_modes(
            X_train_raw=X_train_dataset_raw_np,
            X_train_z  =to_numpy_3d(train_complete_std.X, channels_last=False),
            X_test_raw =X_test_dataset_raw_np,
            X_test_z   =to_numpy_3d(test_complete_std.X,  channels_last=False),
            mu_train   =np.asarray(train_complete_std.mu),   # keep original shapes (1,1,T)
            sd_train   =np.asarray(train_complete_std.sd),
            mode=args['standardization_mode'],
            time_cut_ratio=args['time_cut_ratio'],
            eps=args['eps'], tol_mean=args['tol_mean'], tol_std=args['tol_std']
        )
        ok, msg = standardization_pass_fail_all_modes(report, tol_consistency=1e-4)
        
        print('standardization entire train and test datasets by training set mu and sd',  ok)
        print('standardization entire train and test datasets by training set mu and sd', msg)
        
        
        iterator_train_complete = DataLoader(train_complete_std,
                             batch_size=args['batch_size'],
                             shuffle=True)
        print('\niterator complete training dataset', np.unique((iterator_train_complete.dataset.Y).numpy()))
       
       
        # creates iterator standardized testing dataset (with trajectory and random classes)    (different users)
             

        iterator_test_complete = DataLoader(test_complete_std,
                             batch_size=args['batch_size'],
                             shuffle=True)
      

        print('\niterator complete test dataset', np.unique((iterator_test_complete.dataset.Y).numpy()))
       
       
        # persist standardization in json
      
        
               
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
        
        my_experiment.results["Class info"] = prepare_json_dataset(train_dataset_raw)
        
        gpu_to_use = rank % args["gpus"]
        if torch.cuda.is_available():
            device = torch.device('cuda:' + str(gpu_to_use))
            logger.info("Using gpu : %s", 'cuda:' + str(gpu_to_use))
        else:
            device = torch.device('cpu')
    
        maml = MetaLearingClassification(args, config).to(device)
        

        avg_acc = 0
        acc_loss = []
        loss_iteration = []
 
        
        for step in range(args['steps']): 
            
            print('step ', step)

            t = maml.select_classes2train(classes_trajectory , args['tasks'])
            

            x_spt, y_spt, x_qry, y_qry = maml.select_samples2train_new(dataset_trajectory, t, dataset_random, 
                                                                       classes_random,
                                                                       num_support=args['update_step'], num_query=args['query'], 
                                                                       random=args['random'], reset = args['reset']) 
            '''
            # example augmentation
            augmenter = TimeSeriesAugmenter(time_axis=2, channel_axis=1)
            X_aug, Y_aug = augmenter.augment_dataset(data=x_spt[0], data_augmentation=['Jitter'], Y=y_spt[0], append_to_dataset=False)
           '''
            
            if torch.cuda.is_available():
                x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
                
            accs, loss = maml(x_spt, y_spt, x_qry, y_qry)
            
            avg_acc += accs[-1]
            result_loss = [tensor.item() for tensor in loss]
            loss_iteration.append(result_loss)
                     
                     
            # Evaluation during training for sanity checksi
            if step % 100 == 0 or step ==0:

               #logger.info('step: %d \t training acc %s', step, str(accs))
             
               #result_loss = [tensor.item() for tensor in loss]
               spt = [tensor.item() for tensor in y_spt] 
               results = {"step": step, "acc": accs.tolist(), "loss": result_loss, "y_spt": spt, "y_qry": y_qry.tolist()}
               acc_loss.append(results)
             
        
        if args['replay']:
           print('Generating replay buffer - strategy :',args['replay_strategy'] )

           replay = []
           if args['random']:
              for i in range(0,len(dataset_random.Y)):
                  replay.append([dataset_random.Y[i].item(), dataset_random.X[i].squeeze()])
           for i in range(0,len(dataset_trajectory.Y)):
                replay.append([dataset_trajectory.Y[i].item(), dataset_trajectory.X[i].squeeze()])
           if args['replay_strategy'] is not None: 
              replay_strategies = args['replay_strategy'].split(',') 
              for strategy in replay_strategies:
                  print(strategy)
                  replayBuffer = ReplayBuffer(replay,args['replay_size'], strategy, args['replay_center']) 
                  replayBuffer.save_to_json(my_experiment.path+'replay_'+ strategy +'.json')
                  
        my_experiment.add_result("Learning stats", acc_loss) 

        file_path = my_experiment.path + 'loss.csv'  # Replace with your desired path
        
        
        with open(file_path, "w", newline="") as file_csv:
            writer_csv = csv.writer(file_csv)
            writer_csv.writerows(loss_iteration)

        # to evaluates training with random initialization in PLN 
        # otherwise PLN is initialized with last step
        #maml.net.reset_vars()   
       
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
        
        # running online experiment
        
        if args['online']:  
            gc.collect()
            print("Running online experiment.")
        
            arguments_list = [args['online_file'], "--path", os.path.abspath(my_experiment.path) +'/', "--plot" ]

            if args['replay_online']:
               arguments_list.append("--replay")
            if args['replay_update']:
                arguments_list.append("--replay_update")
            if  args['encoder_update']:   
                arguments_list.append("--encoder_update")
            if args['new_seed']:
               arguments_list.append("--new_seed") 
        
            print('arguments_list', arguments_list)
            try:
                result = subprocess.run(python_command + arguments_list, check=True, capture_output=False, text=True)
                print("Command output:", result.stdout)
            except subprocess.CalledProcessError as e:
                print("Error occurred:", e)
                print("Command output (if available):", e.stdout)
                print("Command error (if available):", e.stderr)
     
if __name__ == '__main__':
    main()
 
