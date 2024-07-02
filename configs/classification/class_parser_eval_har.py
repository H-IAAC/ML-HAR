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
        self.add('--scenario', help= 'nic or nc scenario', default='nc' )

        
        self.add('--plot', help= 'generating plots', action="store_true")
        self.add('--plot_file', help= 'running file for plotting ', type=str, default = 'plot_meta-testing.py')

        


