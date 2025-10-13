import numpy as np

class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_type, dataset, input_dimension=6, output_dimension=6, 
                  width=300, channels=64, data_size='', 
                  cnn_layers=6, kernel = 5, stride =1,
                  out_linear = 100, dropout = 0.4):

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
                 "config": {"out-channels": channels, "in-channels": 1, "kernel": 3, "stride": 2, "padding": 0}},
                {"name": 'relu'},
    
                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernel": 3, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},
    
                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernel": 3, "stride": 2,
                            "padding": 0}},
                {"name": 'relu'},
                #
                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernel": 3, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},
    
                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernel": 3, "stride": 2,
                            "padding": 0}},
                {"name": 'relu'},
    
                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernel": 3, "stride": 2,
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
                 "config": {"out-channels": channels, "in-channels": 9, "kernel": 5, "stride": 1, 
                            "padding": 0}},
                {"name": 'relu'},
            
                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},
                            
                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                           "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                           "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                           "padding": 0}},
                {"name": 'relu'},

                {"name": 'dropout', "adaptation": False, "meta": True,
                 "config": {"p": 0.6}},

                {"name": 'maxPool1d', "adaptation": False, "meta": True,
                 "config": {"kernel": 2,"stride": 2}},
                 
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
                     "config": {"out-channels": channels, "in-channels": data_size[0], "kernel": 5, "stride": 1, 
                                "padding": 0}},
                    {"name": 'relu'},
                
                    {"name": 'conv1d', "adaptation": False, "meta": True,
                     "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                                "padding": 0}},
                    {"name": 'relu'},
                                
                    {"name": 'conv1d', "adaptation": False, "meta": True,
                    "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                               "padding": 0}},
                    {"name": 'relu'},
             
                    {"name": 'conv1d', "adaptation": False, "meta": True,
                    "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                             "padding": 0}},
                    {"name": 'relu'},
    
                    {"name": 'dropout', "adaptation": False, "meta": True,
                     "config": {"p": 0.6}},
    
                    {"name": 'maxPool1d', "adaptation": False, "meta": True,
                     "config": {"kernel": 2,"stride": 2}},
                     
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
                     "config": {"out-channels": channels, "in-channels": data_size[0], "kernel": 5, "stride": 1, 
                                "padding": 0}},
                    {"name": 'relu'},
                
                    {"name": 'conv1d', "adaptation": False, "meta": True,
                     "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                                "padding": 0}},
                    {"name": 'relu'},
                                
                    {"name": 'conv1d', "adaptation": False, "meta": True,
                    "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                               "padding": 0}},
                    {"name": 'relu'},
             
                    {"name": 'conv1d', "adaptation": False, "meta": True,
                    "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                             "padding": 0}},
                    {"name": 'relu'},
                    
                    {"name": 'conv1d', "adaptation": False, "meta": True,
                    "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                             "padding": 0}},
                    {"name": 'relu'},
                    
                    {"name": 'conv1d', "adaptation": False, "meta": True,
                    "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                             "padding": 0}},
                    {"name": 'relu'},
    

                    {"name": 'dropout', "config": {"p": 0.6}},
    
                    {"name": 'maxPool1d', "config": {"kernel": 2,"stride": 2}},
                     
                    {"name": 'flatten'},
                    {"name": 'rep'},
                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"out": out_linear, "in": flatten_out_dim}},
                    
                    {"name": 'relu'},

                    
                    {"name": 'linear', "adaptation": True, "meta": True,
                      "config": {"out": output_dimension, "in": out_linear}}

                ]  
            
        elif dataset == "har_multilayer":
        
            maxpool_out_dim = (data_size[1] - ((kernel * cnn_layers) - (stride * cnn_layers))) // 2
            flatten_out_dim = int(channels * maxpool_out_dim)
            intermediate_dimension1 = (flatten_out_dim + output_dimension) // 2  # First intermediate size
            intermediate_dimension2 = (intermediate_dimension1 + output_dimension) // 2  # Second intermediate size

            
            return [
                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": data_size[0], "kernel": 5, "stride": 1, 
                            "padding": 0}},

                {"name": 'relu'},
            
                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                            "padding": 0}},

                {"name": 'relu'},
                            
                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                           "padding": 0}},
                {"name": 'relu'},
    
                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                           "padding": 0}},

                {"name": 'relu'},
    
                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                            "padding": 0}},

                {"name": 'relu'},
    
                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                           "padding": 0}},

                {"name": 'relu'},
    
                {"name": 'maxPool1d', "adaptation": False, "meta": True,
                 "config": {"kernel": 2,"stride": 2}},
    
                {"name": 'dropout', "adaptation": False, "meta": True,
                 "config": {"p": dropout}},
                 
                {"name": 'flatten'},
                
                {"name": 'rep'},
                    
                {"name": 'linear', "adaptation": True, "meta": True,
                 "config": {"out": intermediate_dimension1, "in": flatten_out_dim}},

                {"name": 'relu'},
                    
                {"name": 'linear', "adaptation": True, "meta": True,
                 "config": {"out": intermediate_dimension2, "in": intermediate_dimension1}},

                {"name": 'relu'},
                
                {"name": 'linear', "adaptation": True, "meta": True,
                  "config": {"out": output_dimension, "in": intermediate_dimension2}}
           
            ]    
        
        elif dataset == "har_1layer":
        
            maxpool_out_dim = (data_size[1] - ((kernel * cnn_layers) - (stride * cnn_layers))) // 2
            flatten_out_dim = int(channels * maxpool_out_dim)
             
            return [
                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": data_size[0], "kernel": 5, "stride": 1, 
                            "padding": 0}},
               
                {"name": 'relu'},
            
                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                            "padding": 0}},
                
                {"name": 'relu'},
                            
                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                           "padding": 0}},
                
                {"name": 'relu'},
    
                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                           "padding": 0}},
                
                {"name": 'relu'},
    
                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                            "padding": 0}},
                
                {"name": 'relu'},
    
                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                           "padding": 0}},
                {"name": 'relu'},
    
                {"name": 'maxPool1d', "adaptation": False, "meta": True,
                 "config": {"kernel": 2,"stride": 2}},
    
                {"name": 'dropout', "adaptation": False, "meta": True,
                 "config": {"p": dropout}},
                 
                {"name": 'flatten'},
                
                {"name": 'rep'},
                
                {"name": 'linear', "adaptation": True, "meta": True,
                  "config": {"out": output_dimension, "in": flatten_out_dim}}
           
            ]         
        
       
        elif dataset == "har_lstm_RLN_2layers":

            lstm_hidden_size = 128
            
            maxpool_out_dim = (data_size[1] - ((kernel * cnn_layers) - (stride * cnn_layers))) // 2
            
            flatten_out_dim = lstm_hidden_size * channels
            
            intermediate_dimension1 = (flatten_out_dim + output_dimension) // 2  # First intermediate size
            intermediate_dimension2 = (intermediate_dimension1 + output_dimension) // 2  # Second intermediate size



            return [
                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": data_size[0], "kernel": 5, "stride": 1, 
                            "padding": 0}},
                
            
                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},
                            
                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                           "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                           "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                           "padding": 0}},
                {"name": 'relu'},

                {"name": 'maxPool1d', "adaptation": False, "meta": True,
                 "config": {"kernel": 2,"stride": 2}},

                {"name": 'dropout', "adaptation": False, "meta": True,
                 "config": {"p": dropout}},
                
                # LSTM layers
                {"name": 'lstm', "adaptation": False, "meta": True,
                 "config": {"hidden_size": lstm_hidden_size, "input_size": maxpool_out_dim, "num_layers": 1,
                             "batch_first": True}},
                
                {"name": 'lstm', "adaptation": False, "meta": True,
                 "config": {"hidden_size": lstm_hidden_size, "input_size": lstm_hidden_size, "num_layers": 1,
                             "batch_first": True}},                
                 
                {"name": 'flatten'},
                
                {"name": 'rep'},
                                
                {"name": 'linear', "adaptation": True, "meta": True,
                 "config": {"out": intermediate_dimension1, "in": flatten_out_dim}},
                {"name": 'relu'},
                    
                {"name": 'linear', "adaptation": True, "meta": True,
                 "config": {"out": intermediate_dimension2, "in": intermediate_dimension1}},

                {"name": 'relu'},
                
                {"name": 'linear', "adaptation": True, "meta": True,
                  "config": {"out": output_dimension, "in": intermediate_dimension2}}
           
            ]
            
        elif dataset == "har_lstm_RLN_1layer":

            lstm_hidden_size = 128
            
            maxpool_out_dim = (data_size[1] - ((kernel * cnn_layers) - (stride * cnn_layers))) // 2
            
            flatten_out_dim = lstm_hidden_size * channels
            
            intermediate_dimension1 = (flatten_out_dim + output_dimension) // 2  # First intermediate size
            intermediate_dimension2 = (intermediate_dimension1 + output_dimension) // 2  # Second intermediate size



            return [
                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": data_size[0], "kernel": 5, "stride": 1, 
                            "padding": 0}},
                
            
                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},
                            
                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                           "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                           "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                            "padding": 0}},

                {"name": 'relu'},

                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                           "padding": 0}},
                {"name": 'relu'},

                {"name": 'maxPool1d', "adaptation": False, "meta": True,
                 "config": {"kernel": 2,"stride": 2}},

                {"name": 'dropout', "adaptation": False, "meta": True,
                 "config": {"p": dropout}},
                
                # LSTM layers
                {"name": 'lstm', "adaptation": False, "meta": True,
                 "config": {"hidden_size": lstm_hidden_size, "input_size": maxpool_out_dim, "num_layers": 1,
                             "batch_first": True}},
                 
                {"name": 'flatten'},
                
                {"name": 'rep'},
                                
                {"name": 'linear', "adaptation": True, "meta": True,
                 "config": {"out": intermediate_dimension1, "in": flatten_out_dim}},
                {"name": 'relu'},
                    
                {"name": 'linear', "adaptation": True, "meta": True,
                 "config": {"out": intermediate_dimension2, "in": intermediate_dimension1}},

                {"name": 'relu'},
                
                {"name": 'linear', "adaptation": True, "meta": True,
                  "config": {"out": output_dimension, "in": intermediate_dimension2}}
           
            ]            
        
        elif dataset == "har_lstm_2layers":
            
            lstm_hidden_size = 128
            
            maxpool_out_dim = (data_size[1] - ((kernel * cnn_layers) - (stride * cnn_layers))) // 2
            
            flatten_out_dim = lstm_hidden_size * channels
            
            intermediate_dimension1 = (flatten_out_dim + output_dimension) // 2  # First intermediate size
            intermediate_dimension2 = (intermediate_dimension1 + output_dimension) // 2  # Second intermediate size


            return [
                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": data_size[0], "kernel": 5, "stride": 1, 
                            "padding": 0}},
                
            
                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},
                            
                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                           "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                           "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                           "padding": 0}},
                {"name": 'relu'},

                {"name": 'maxPool1d', "adaptation": False, "meta": True,
                 "config": {"kernel": 2,"stride": 2}},

                {"name": 'dropout', "adaptation": False, "meta": True,
                 "config": {"p": dropout}},
                
                                
                
                # LSTM layers RLN
                {"name": 'lstm', "adaptation": False, "meta": True,
                 "config": {"hidden_size": lstm_hidden_size, "input_size": maxpool_out_dim, "num_layers": 1,
                             "batch_first": True}},
                
                {"name": 'lstm', "adaptation": False, "meta": True,
                 "config": {"hidden_size": lstm_hidden_size, "input_size": lstm_hidden_size, "num_layers": 1,
                             "batch_first": True}},    
                
                
                # LSTM layers PLN
                {"name": 'lstm', "adaptation": True, "meta": True,
                 "config": {"hidden_size": lstm_hidden_size, "input_size": lstm_hidden_size, "num_layers": 1,
                             "batch_first": True}},
                
                {"name": 'lstm', "adaptation": True, "meta": True,
                 "config": {"hidden_size": lstm_hidden_size, "input_size": lstm_hidden_size, "num_layers": 1,
                             "batch_first": True}},      
                 
                {"name": 'flatten'},
                
                {"name": 'rep'},
                                
                {"name": 'linear', "adaptation": True, "meta": True,
                 "config": {"out": intermediate_dimension1, "in": flatten_out_dim}},

                {"name": 'relu'},
                    
                {"name": 'linear', "adaptation": True, "meta": True,
                 "config": {"out": intermediate_dimension2, "in": intermediate_dimension1}},
                {"name": 'relu'},
                
                {"name": 'linear', "adaptation": True, "meta": True,
                  "config": {"out": output_dimension, "in": intermediate_dimension2}}
           
            ]
            
        elif dataset == "har_lstm_1layer":
            
            lstm_hidden_size = 128
            
            maxpool_out_dim = (data_size[1] - ((kernel * cnn_layers) - (stride * cnn_layers))) // 2
            
            flatten_out_dim = lstm_hidden_size * channels
            
            intermediate_dimension1 = (flatten_out_dim + output_dimension) // 2  # First intermediate size
            intermediate_dimension2 = (intermediate_dimension1 + output_dimension) // 2  # Second intermediate size


            return [
                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": data_size[0], "kernel": 5, "stride": 1, 
                            "padding": 0}},
                
            
                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},
                            
                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                           "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                           "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                           "padding": 0}},
                {"name": 'relu'},

                {"name": 'maxPool1d', "adaptation": False, "meta": True,
                 "config": {"kernel": 2,"stride": 2}},

                {"name": 'dropout', "adaptation": False, "meta": True,
                 "config": {"p": dropout}},
                
                                
                
                # LSTM layers RLN
                {"name": 'lstm', "adaptation": False, "meta": True,
                 "config": {"hidden_size": lstm_hidden_size, "input_size": maxpool_out_dim, "num_layers": 1,
                             "batch_first": True}},
                
                
                # LSTM layers PLN
                {"name": 'lstm', "adaptation": True, "meta": True,
                 "config": {"hidden_size": lstm_hidden_size, "input_size": lstm_hidden_size, "num_layers": 1,
                             "batch_first": True}},
                
                 
                {"name": 'flatten'},
                
                {"name": 'rep'},
                                
                {"name": 'linear', "adaptation": True, "meta": True,
                 "config": {"out": intermediate_dimension1, "in": flatten_out_dim}},

                {"name": 'relu'},
                    
                {"name": 'linear', "adaptation": True, "meta": True,
                 "config": {"out": intermediate_dimension2, "in": intermediate_dimension1}},
                {"name": 'relu'},
                
                {"name": 'linear', "adaptation": True, "meta": True,
                  "config": {"out": output_dimension, "in": intermediate_dimension2}}
           
            ]            
        elif dataset == "har_lstm_PLN_2layers":
            
            lstm_hidden_size = 128
            
            maxpool_out_dim = (data_size[1] - ((kernel * cnn_layers) - (stride * cnn_layers))) // 2
            
            flatten_out_dim = lstm_hidden_size * channels
            
            intermediate_dimension1 = (flatten_out_dim + output_dimension) // 2  # First intermediate size
            intermediate_dimension2 = (intermediate_dimension1 + output_dimension) // 2  # Second intermediate size


            return [
                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": data_size[0], "kernel": 5, "stride": 1, 
                            "padding": 0}},
            
                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},
                            
                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                           "padding": 0}},

                {"name": 'relu'},

                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                           "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                           "padding": 0}},
                {"name": 'relu'},

                {"name": 'maxPool1d', "adaptation": False, "meta": True,
                 "config": {"kernel": 2,"stride": 2}},

                {"name": 'dropout', "adaptation": False, "meta": True,
                 "config": {"p": dropout}},
                
                # LSTM layers
                {"name": 'lstm', "adaptation": True, "meta": True,
                 "config": {"hidden_size": lstm_hidden_size, "input_size": maxpool_out_dim, "num_layers": 1,
                             "batch_first": True}},
                
                {"name": 'lstm', "adaptation": True, "meta": True,
                 "config": {"hidden_size": lstm_hidden_size, "input_size": lstm_hidden_size, "num_layers": 1,
                             "batch_first": True}},                
                 
                {"name": 'flatten'},
                
                {"name": 'rep'},
                                
                {"name": 'linear', "adaptation": True, "meta": True,
                 "config": {"out": intermediate_dimension1, "in": flatten_out_dim}},

                {"name": 'relu'},
                    
                {"name": 'linear', "adaptation": True, "meta": True,
                 "config": {"out": intermediate_dimension2, "in": intermediate_dimension1}},
                {"name": 'relu'},
                
                {"name": 'linear', "adaptation": True, "meta": True,
                  "config": {"out": output_dimension, "in": intermediate_dimension2}}
           
            ]
        elif dataset == "har_lstm_PLN_1layer":
            
            lstm_hidden_size = 128
            
            maxpool_out_dim = (data_size[1] - ((kernel * cnn_layers) - (stride * cnn_layers))) // 2
            
            flatten_out_dim = lstm_hidden_size * channels
            
            intermediate_dimension1 = (flatten_out_dim + output_dimension) // 2  # First intermediate size
            intermediate_dimension2 = (intermediate_dimension1 + output_dimension) // 2  # Second intermediate size


            return [
                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": data_size[0], "kernel": 5, "stride": 1, 
                            "padding": 0}},
            
                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},
                            
                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                           "padding": 0}},

                {"name": 'relu'},

                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                           "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv1d', "adaptation": False, "meta": True,
                "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv1d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernel": 5, "stride": 1,
                           "padding": 0}},
                {"name": 'relu'},

                {"name": 'maxPool1d', "adaptation": False, "meta": True,
                 "config": {"kernel": 2,"stride": 2}},

                {"name": 'dropout', "adaptation": False, "meta": True,
                 "config": {"p": dropout}},
                
                # LSTM layers
                {"name": 'lstm', "adaptation": True, "meta": True,
                 "config": {"hidden_size": lstm_hidden_size, "input_size": maxpool_out_dim, "num_layers": 1,
                             "batch_first": True}},
                
                {"name": 'flatten'},
                
                {"name": 'rep'},
                                
                {"name": 'linear', "adaptation": True, "meta": True,
                 "config": {"out": intermediate_dimension1, "in": flatten_out_dim}},

                {"name": 'relu'},
                    
                {"name": 'linear', "adaptation": True, "meta": True,
                 "config": {"out": intermediate_dimension2, "in": intermediate_dimension1}},
                {"name": 'relu'},
                
                {"name": 'linear', "adaptation": True, "meta": True,
                  "config": {"out": output_dimension, "in": intermediate_dimension2}}
           
            ]           
        else:
            print("Unsupported model; either implement the model in model/ModelFactory or choose a different model")
            assert (False)

