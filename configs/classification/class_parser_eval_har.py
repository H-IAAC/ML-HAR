import configargparse


class Parser(configargparse.ArgParser):
    def __init__(self):
        super().__init__()
        self.add('-c', '--my-config', is_config_file=True, default="configs/regression/empty.ini",
                 help='config file path')
        #
        self.add('--steps', type=int, help='epoch number', default=10)
        self.add('--gpus', type=int, help='meta-level outer learning rate', default=1)
        self.add('--rank', type=int, help='meta batch size, namely task num', default=0)
        self.add('--tasks', nargs='+', type=int, help='meta batch size, namely task num', default=[2])
        self.add('--scratch', help= 'run metatest from scratch', action="store_true")
        self.add('--folder_id', help= 'folder to persist results', default="../results/teste/")

        self.add('--meta_lr', nargs='+', type=float, help='meta-level outer learning rate', default=[1e-4])
        self.add('--update_lr', nargs='+', type=float, help='task-level inner update learning rate', default=[0.01])
        self.add('--update_step', nargs='+', type=int, help='task-level inner update steps', default=[10])
        self.add('--query', help='number of query samples',default=[5] )
        self.add('--dataset', help='Name of experiment', default="ucihar")
        self.add('--seed', nargs='+', help='Seed', default=[90], type=int)
        self.add("--new_seed",help='datetime seed', action="store_true")
        self.add('--name', help='Name of experiment', default="online")
        self.add('--path', help='Path of the dataset', default="../")
        self.add('--batch_size', help='batch size learning', default=[20], type=int)
        self.add('--json_config', help='json config to stats generation',default="/configs/online_stats.json" )
        
        
        

        self.add('--schedule', type=str, nargs='+', default="10",
                               help='Decrease learning rate at these epochs.')
        self.add('--classes_schedule', type=int, default=2,
                               help='Number of classes to schedule.')       
        self.add('--reset_weights', action="store_true")
        self.add('--test', action="store_true")
        self.add("--iid", action="store_true")
        self.add("--runs", type=int, default=5)
        self.add('--model-path', nargs='+', type=str, help='path to trained model', default=None)
        self.add('--model', nargs='+', type=str, help='model id: maml, oml, proto', default='oml')
        self.add('--dataset_path', nargs='+', type=str, help='root path to dataset files', default=None)
        self.add('--scenario', help= 'nic or nc scenario', default='nic' )
        self.add('--replay', help= 'use of replay memorey', action="store_true")
        self.add('--replay_update', help= 'update replay online', action="store_true")
        self.add('--replay_strategy',  help= 'replay strategy', type=str, default = None)
        self.add('--encoder_update', help= 'update encoder online', action="store_true")
        self.add('--encoder_replay', help= 'update encoder online using replay data', action="store_true")
        
        self.add('--encoder_strategy', help= 'when encoder will be updated with replay or all', default="replay")
        self.add('--encoder_classes', help= 'which classes using to encoder update updated replay or all', default="all")     

        self.add('--encoder_linear', help= 'update weights', action="store_true")
        self.add('--encoder_ML', help= 'only ML', action="store_true")
        self.add('--only_ML', help= 'only ML', action="store_true")
        
        self.add('--plot', help= 'generating plots', action="store_true")
        self.add('--plot_file', help= 'running file for plotting ', type=str, default = 'plot_online.py')

        

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
        self.add('--lr_decreasing_factor', help= 'lr decreasing factor', type=int, default = 0.01)
        self.add('--norm_explosion_threshold', nargs='+', type=float, help='gradient norm value limit', default=[1e2])   
