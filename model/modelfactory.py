import numpy as np

class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_type, dataset, input_dimension=6, output_dimension=6, 
                  width=300, channels=64, data_size='', 
                  cnn_layers=6, kernel = 5, stride =1,
                  out_linear = 100):

        if "Sin" == dataset:

            if model_type == "representation":

                hidden_size = width
                return [

                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"out": hidden_size, "in": input_dimension}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": True, "meta": True,
                     "config": {"out": output_dimension, "in": hidden_size}}
                ]

        elif dataset == "omniglot":
            channels = 256
    
            return [
                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": 1, "kernal": 3, "stride": 2, "padding": 0}},
                {"name": 'relu'},
    
                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},
    
                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
                            "padding": 0}},
                {"name": 'relu'},
                #
                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},
    
                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
                            "padding": 0}},
                {"name": 'relu'},
    
                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
                            "padding": 0}},
                {"name": 'relu'},
    
                {"name": 'flatten'},

                {"name": 'rep'},
    
                {"name": 'linear', "adaptation": True, "meta": True,
                 "config": {"out": 1000, "in": 9 * channels}}
    
            ]

        elif dataset == "ucihar":
            channels = 64
            # 2 layers - linear in = 3840
            # 4 layers - linear in = 3584
            # 6 layers - linear in = 3328
            return [

                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": 9, "kernal": 5, "stride": 1, 
                            "padding": 0}},
                {"name": 'relu'},
            
                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 5, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},
                            
                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernal": 5, "stride": 1,
                           "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernal": 5, "stride": 1,
                           "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernal": 5, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 5, "stride": 1,
                           "padding": 0}},
                {"name": 'relu'},

                {"name": 'dropout', "adaptation": False, "meta": True,
                 "config": {"p": 0.6}},

                {"name": 'maxPool1d', "adaptation": False, "meta": True,
                 "config": {"kernal": 2,"stride": 2}},
                 
                {"name": 'flatten'},
                {"name": 'rep'},
                {"name": 'linear', "adaptation": False, "meta": True,
                 "config": {"out": 100, "in": 3328}},
                
                {"name": 'relu'},
                
                {"name": 'linear', "adaptation": True, "meta": True,
                 "config": {"out": 6, "in": 100}},
           
            ]
        
        elif dataset == "har":
            
 
            maxpool_out_dim = (data_size[1] - ((kernel * cnn_layers) - (stride * cnn_layers))) // 2
            flatten_out_dim = int(channels * maxpool_out_dim)
                       
            if cnn_layers == 4:
           
                return [  
                    {"name": 'guassianNoise', "adaptation": False, "meta": True,
                     "config": {"std": 0.1}},
                
                    {"name": 'conv1d', "adaptation": False, "meta": True,
                     "config": {"out-channels": channels, "in-channels": data_size[0], "kernal": 5, "stride": 1, 
                                "padding": 0}},
                    {"name": 'relu'},
                
                    {"name": 'conv1d', "adaptation": False, "meta": True,
                     "config": {"out-channels": channels, "in-channels": channels, "kernal": 5, "stride": 1,
                                "padding": 0}},
                    {"name": 'relu'},
                                
                    {"name": 'conv1d', "adaptation": False, "meta": True,
                    "config": {"out-channels": channels, "in-channels": channels, "kernal": 5, "stride": 1,
                               "padding": 0}},
                    {"name": 'relu'},
             
                    {"name": 'conv1d', "adaptation": False, "meta": True,
                    "config": {"out-channels": channels, "in-channels": channels, "kernal": 5, "stride": 1,
                             "padding": 0}},
                    {"name": 'relu'},
    
                    {"name": 'dropout', "adaptation": False, "meta": True,
                     "config": {"p": 0.6}},
    
                    {"name": 'maxPool1d', "adaptation": False, "meta": True,
                     "config": {"kernal": 2,"stride": 2}},
                     
                    {"name": 'flatten'},
                    {"name": 'rep'},
                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"out": out_linear, "in": flatten_out_dim}},
                    
                    {"name": 'relu'},
                    
                    {"name": 'linear', "adaptation": True, "meta": True,
                     "config": {"out": output_dimension, "in": out_linear}},
              
                ]
            else:
                return [      
                
                    {"name": 'conv1d', "adaptation": False, "meta": True,
                     "config": {"out-channels": channels, "in-channels": data_size[0], "kernal": 5, "stride": 1, 
                                "padding": 0}},
                    {"name": 'relu'},
                
                    {"name": 'conv1d', "adaptation": False, "meta": True,
                     "config": {"out-channels": channels, "in-channels": channels, "kernal": 5, "stride": 1,
                                "padding": 0}},
                    {"name": 'relu'},
                                
                    {"name": 'conv1d', "adaptation": False, "meta": True,
                    "config": {"out-channels": channels, "in-channels": channels, "kernal": 5, "stride": 1,
                               "padding": 0}},
                    {"name": 'relu'},
             
                    {"name": 'conv1d', "adaptation": False, "meta": True,
                    "config": {"out-channels": channels, "in-channels": channels, "kernal": 5, "stride": 1,
                             "padding": 0}},
                    {"name": 'relu'},
                    
                    {"name": 'conv1d', "adaptation": False, "meta": True,
                    "config": {"out-channels": channels, "in-channels": channels, "kernal": 5, "stride": 1,
                             "padding": 0}},
                    {"name": 'relu'},
                    
                    {"name": 'conv1d', "adaptation": False, "meta": True,
                    "config": {"out-channels": channels, "in-channels": channels, "kernal": 5, "stride": 1,
                             "padding": 0}},
                    {"name": 'relu'},
    

                    {"name": 'dropout', "config": {"p": 0.6}},
    
                    {"name": 'maxPool1d', "config": {"kernal": 2,"stride": 2}},
                     
                    {"name": 'flatten'},
                    {"name": 'rep'},
                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"out": out_linear, "in": flatten_out_dim}},
                    
                    {"name": 'relu'},

                    
                    {"name": 'linear', "adaptation": True, "meta": True,
                      "config": {"out": output_dimension, "in": out_linear}}

                ]       
       
        elif dataset == "har_1layer":
            
            maxpool_out_dim = (data_size[1] - ((kernel * cnn_layers) - (stride * cnn_layers))) // 2
            flatten_out_dim = int(channels * maxpool_out_dim)

            return [
                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": data_size[0], "kernal": 5, "stride": 1, 
                            "padding": 0}},
                {"name": 'relu'},
            
                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 5, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},
                            
                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernal": 5, "stride": 1,
                           "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernal": 5, "stride": 1,
                           "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernal": 5, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 5, "stride": 1,
                           "padding": 0}},
                {"name": 'relu'},

                {"name": 'maxPool1d', "adaptation": False, "meta": True,
                 "config": {"kernal": 2,"stride": 2}},

                {"name": 'dropout', "adaptation": False, "meta": True,
                 "config": {"p": 0.6}},
                 
                {"name": 'flatten'},
                
                {"name": 'rep'},
                
                {"name": 'linear', "adaptation": True, "meta": True,
                  "config": {"out": output_dimension, "in": flatten_out_dim}}
           
            ]
        else:
            print("Unsupported model; either implement the model in model/ModelFactory or choose a different model")
            assert (False)
