import configargparse


class Parser(configargparse.ArgParser):
    def __init__(self):
        super().__init__()
        self.add('-c', '--my-config', is_config_file=True, default="configs/regression/empty.ini",
                 help='config file path')

        self.add('--gpus', type=int, help='meta-level outer learning rate', default=1)
        self.add('--rank', type=int, help='meta batch size, namely task num', default=0)
        self.add("--runs", type=int, default=5)
        self.add('--steps', type=int, help='epoch number', default=10000)
        self.add('--lr', nargs='+', type=float, help='task-level inner update learning rate', default=[5e-4])  #1e-5 5e-4 5e-05
        self.add('--folder_id', help= 'folder to persist results', default="../results/encoders/")
        
        self.add('--seed', nargs='+', help='Seed', default=[90], type=int)
        self.add('--name', help='Name of experiment', default="baseline")
        self.add('--scenario', help= 'scenario', default='nc' )
        self.add('--fraction_subject', help='Fraction of subjects for split train into online and offline train',default=0.6, type=float)
        self.add('--fraction_subject_validation', help='Fraction of subjects for split train into val and train sets',default=0.3, type=float)
        self.add('--decay', help='If weight decay',  action="store_true")
        self.add("--new_seed",help='datetime seed', action="store_true")
        self.add('--schedule', type=int, nargs='+', default=50,
                               help='Decrease learning rate at these epochs.')
        self.add('--l2', help='If regularization',  action="store_true")
        self.add('--decay_factor', type=float, help='learning rate decay.', default=[1.00E-04])
        self.add('--l2_lambda', type=float, help='learning rate decay.', default=[0.01])
        self.add('--reset',help='Reset weights.',default=False)
        self.add("--iid", action="store_true",default=True)
        self.add('--dataset', help='Name of dataset', default="pamap2")
        self.add('--dataset_path', nargs='+', type=str, help='root path to dataset files', default=None)
        self.add('--path', help='Path of the dataset', default="../")
        self.add('--batch_size', help='batch size learning', default=[32], type=int)
        self.add('--is_standardized', help='Standarization of datasets',  default=True)
        self.add('--channels', help= 'number of channels in conv1d', type=int, default = 64)
        self.add('--layers', help= 'number of conv1d layers', type=int, default = 6)
        self.add('--kernel', help= 'number of kernel layers', type=int, default = 5)
        self.add('--stride', help= 'stride of layers', type=int, default = 1)
        self.add('--out_linear', help= 'out_features dimension linear layer', type=int, default = 100)

        self.add('--plot', help= 'generating plots',action="store_true")
        self.add('--plot_file', help= 'running file for plotting ', type=str, default = 'plot_meta-testing_batch.py')
        
        self.add('--fraction_classes', help='Fraction of classes for offline training',default=0.6, type=float)
        
        self.add('--augmentation', help= 'augmentarion technique', default=None)
 

        


