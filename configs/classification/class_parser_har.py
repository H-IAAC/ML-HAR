import configargparse


class Parser(configargparse.ArgParser):
    def __init__(self):
        super().__init__()
        self.add('-c', '--my-config', is_config_file=True, default="configs/regression/empty.ini",
                 help='config file path')

        self.add('--main_folder', help='name main experiment results folder',  default = "results") 
        self.add('--steps', type=int, help='epoch number', default=1000)
        self.add('--runs', type=int, help='number of runs', default=1)
        self.add('--iterations', type=int, help='number of iterations - simulation runs', default=1)
        self.add('--gpus', type=int, help='meta-level outer learning rate', default=1)
        self.add('--rank', type=int, help='meta batch size, namely task num', default=0)
        self.add('--tasks', nargs='+', type=int, help='meta batch size, namely task num', default=[2])
        self.add('--seed', nargs='+', help='Seed', default=[90], type=int)
        self.add("--new_seed",help='datetime seed', action="store_true")
        self.add('--name', help='Name of experiment', default="encoders")
        self.add('--model', help='Name of experiment', default="maml")
        
        self.add('--meta_lr', nargs='+', type=float, help='meta-level outer learning rate', default= [5e-4])  #0.0001
        self.add('--update_lr', nargs='+', type=float, help='task-level inner update learning rate', default=[0.01])
        self.add('--update_step', nargs='+', type=int, help='task-level inner update steps', default=[10])


        # benchmarks generatation 
        self.add('--batch_size', help='Train and Test batch size',default=20 )
        self.add('--is_standardized', help='Standarization of datasets',  default=True)
        self.add('--fraction_classes', help='Fraction of classes for offline training',default=0.6, type=float)
        self.add('--fraction_subject', help='Fraction of subjects for offline training nic scenarios',default=0.5, type=float)
        self.add('--dataset', help='Name of experiment', default="pamap2")
        self.add("--reset",help='reset weights steps', action="store_true")
        self.add('--dataset_path', help='Path of the dataset', default=None)
        self.add('--scenario', help= 'nic or nc scenario', default='nc' )
        self.add('--augmentation', help= 'augmentarion technique', default=None)
        self.add('--random', help= 'considers random in query sample', action="store_true")
        self.add('--query', help='number of query samples',default=[10] )
      


        #neural network
        self.add('--channels', help= 'number of channels in conv1d', type=int, default = 64)
        self.add('--layers', help= 'number of conv1d layers', type=int, default = 6)
        self.add('--kernel', help= 'number of kernel layers', type=int, default = 5)
        self.add('--stride', help= 'stride of layers', type=int, default = 1)
        self.add('--out_linear', help= 'out_features dimension linear layer', type=int, default = 100)
        self.add('--network_id', help= 'id to model factory generates neural architecture', type=str, default = 'har_1layer')
        
        self.add('--plot', help= 'generating plots', action="store_true")
        self.add('--plot_file', help= 'running file for plotting ', type=str, default = 'plot_encoder.py')
        

        
        
 

