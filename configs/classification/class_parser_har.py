import configargparse


class Parser(configargparse.ArgParser):
    def __init__(self):
        super().__init__()
        self.add('-c', '--my-config', is_config_file=True, default="configs/regression/empty.ini",
                 help='config file path')

        #self.add('--main_folder', help='name main experiment results folder',  default = "results_gradient") 
        self.add('--main_folder', help='name main experiment results folder',  default = "new_protocol") 
        self.add('--steps', type=int, help='epoch number', default=1000)
        self.add('--runs', type=int, help='number of runs', default=1)
        self.add('--iterations', type=int, help='number of iterations - simulation runs', default=1)
        self.add('--gpus', type=int, help='meta-level outer learning rate', default=1)
        self.add('--rank', type=int, help='meta batch size, namely task num', default=0)
        self.add('--tasks', nargs='+', type=int, help='meta batch size, namely task num', default=[2])
        self.add('--seed', nargs='+', help='Seed', default=[90], type=int)
        self.add("--new_seed",help='datetime seed', action="store_true")
        self.add('--name', help='Name of experiment', default="encoders")
        self.add('--model', help='Name of experiment', default="metagcd") #maml, oml, metagcd
                
        self.add('--meta_lr', nargs='+', type=float, help='meta-level outer learning rate', default=[5e-4]) # [1e-3]) #[5e-4])  #0.0001
        self.add('--update_lr', nargs='+', type=float, help='task-level inner update learning rate', default=[0.001])
        self.add('--update_step', nargs='+', type=int, help='task-level inner update steps', default=[10]) # trajectory


        # benchmarks generatation 
        self.add('--batch_size', help='Train and Test batch size',default=20 )
        self.add('--is_standardized', help='Standarization of datasets',  default=True)
        self.add('--fraction_classes', help='Fraction of classes for offline training',default=0.6, type=float)
        self.add('--fraction_subject', help='Fraction of subjects for offline training nic scenarios',default=0.5, type=float)
        self.add('--dataset', help='Name of experiment', default="dsads")
        self.add("--reset",help='reset weights steps', action="store_true")
        self.add('--dataset_path', help='Path of the dataset', default=None)
        self.add('--scenario', help= 'nic or nc scenario', default='nc' )
        self.add('--augmentation', help= 'augmentarion technique', default=None)
        self.add('--random', help= 'considers random in query sample', action="store_true")
        self.add('--query', help='number of query samples',default=[10] )
      

        # memory replay generation configuration
        self.add('--replay', help= 'generates replay set', action="store_true" )
        self.add('--replay_online', help= 'if use replay in online',  action="store_true")
        self.add('--replay_update', help= 'if update replay in online',  action="store_true")
        self.add('--replay_strategy',  help= 'replay strategy', type=str, default = None)
        self.add('--replay_size', help= 'number of samples per class', default=6)
        self.add('--replay_center',  help= 'replay strategy', type=str, default = 'mean')

        #neural network
        self.add('--channels', help= 'number of channels in conv1d', type=int, default = 64)
        self.add('--layers', help= 'number of conv1d layers', type=int, default = 6)
        self.add('--kernel', help= 'number of kernel layers', type=int, default = 5)
        self.add('--stride', help= 'stride of layers', type=int, default = 1)
        self.add('--out_linear', help= 'out_features dimension linear layer', type=int, default = 100)
        self.add('--network_id', help= 'id to model factory generates neural architecture', type=str, default = 'har_1layer')
        
        #sub processes execution
        self.add('--online', help= 'run online experiment', action="store_true" )
        self.add('--online_file', help= 'run online experiment', type=str, default = 'online_OML.py')
        self.add('--plot', help= 'generating plots', action="store_true")
        self.add('--plot_file', help= 'running file for plotting ', type=str, default = 'plot_encoder.py')
        

         #stop criteria
         
        self.add('--stop_training', help= 'patience criteria ',action="store_true")
        self.add('--stop_criteria', help= 'stop criteria [average = norm average or iteration = number of iteration without changes/', type=str, default='average')
        self.add('--patience_threshold', help= 'patience threshold', type=int, default = 15)
        self.add('--grad_norm_threshold', nargs='+', type=float, help='gradient norm value limit', default=[1e-5])
        self.add('--grad_norm_change', nargs='+', type=float, help='gradient norm change', default=[1e-1])
        self.add('--clip',help= 'if clip gradient ',action="store_true")
        self.add('--clip_value', nargs='+', type=float, help='clip value', default=20.0)
         
        self.add('--clip_inner',help= 'if clip gradient ',action="store_true")
        self.add('--clip_outer',help= 'if clip gradient ',action="store_true")
        self.add('--adaptive_clip',help= 'if clip value is adaptive',action="store_true")
        

        self.add('--lr_evaluation', help= 'evaluate reducing lr ',action="store_true")
        self.add('--lr_evaluation_step', help= 'step to evaluate reducing lr', type=int, default = 30)
        self.add('--lr_decreasing_factor', help= 'lr decreasing factor', type=float, default = 0.01)
        self.add('--norm_explosion_threshold', nargs='+', type=float, help='gradient norm value limit', default=[1e2])       
        
        #standardization 
        
        self.add('--standardization_mode', help= 'standardization mode', type=str, default = 'channel') # channel, timestep, timestep_pooled, window_samplewise
        self.add('--time_cut_ratio', help= 'lr decreasing factor', type=int, default = 0.90) # for channel only
        self.add('--eps', help= 'lr decreasing factor', type=float, default = 1e-8) 
        self.add('--tol_mean', help= 'lr decreasing factor', type=float, default = 0.5)
        self.add('--tol_std', help= 'lr decreasing factor', type=float, default = 0.5)
        

