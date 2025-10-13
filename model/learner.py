import logging

logger = logging.getLogger("experiment")

import logging

import torch
import copy

from torch import nn
from torch.nn import functional as F
import oml
logger = logging.getLogger("experiment")


class Learner(nn.Module):
    """
    """

    def __init__(self, learner_configuration, backbone_configuration=None):
        """

        :param learner_configuration: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__()

        self.config = learner_configuration
        self.backbone_config = backbone_configuration

        self.vars = nn.ParameterList()

        self.vars = self.parse_config(self.config, nn.ParameterList())
        self.context_backbone = None

    def parse_config(self, config, vars_list):

        

        for i, info_dict in enumerate(config):
   
            if info_dict["name"] == 'conv2d':
                w, b = oml.nn.conv2d(info_dict["config"], info_dict["adaptation"], info_dict["meta"])
                vars_list.append(w)
                vars_list.append(b)

           
            elif info_dict["name"] == 'conv1d':
                w, b = oml.nn.conv1d(info_dict["name"],info_dict["config"], info_dict["adaptation"], info_dict["meta"])
                vars_list.append(w)
                vars_list.append(b) 
               
            elif info_dict["name"] == 'lstm':
                lstm_params = oml.nn.LSTM(info_dict["config"], info_dict["adaptation"], info_dict["meta"])
                #vars_list.append(w)
                #vars_list.append(b) 
                vars_list.extend(lstm_params)  # Add all LSTM parameters directly

 
            elif info_dict["name"] == 'linear':
                param_config = info_dict["config"]
                w, b = oml.nn.linear(info_dict["name"],param_config["out"], param_config["in"], info_dict["adaptation"], info_dict["meta"])
 
                vars_list.append(w)
                vars_list.append(b)

            elif info_dict["name"] == 'batchNorm1d':
                param_config = info_dict["config"]
                bn_params = oml.nn.batchNorm1d(param_config["num_features"])
                vars_list.extend(bn_params) 

            elif info_dict["name"] in ['tanh', 'rep', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d', 'maxPool1d', 'dropout',
                                       'flatten', 'reshape', 'leakyrelu', 'sigmoid', 'rotate']:
                continue
            else:
                print(info_dict["name"])
                raise NotImplementedError
        return vars_list

    def add_rotation(self):
        self.rotate = nn.Parameter(torch.ones(2304,2304))
        torch.nn.init.uniform_(self.rotate)
        self.rotate_inverse = nn.Parameter(torch.inverse(self.rotate))
        # #print(self.rotate.shape)
        # #print(self.rotate_inverse.shape)
        # quit()
        logger.info("Inverse computed")

    '''
    def reset_vars(self):
        """
        Reset all adaptation parameters to random values. Bias terms are set to zero and other terms to default values of kaiming_normal_
        :return:
        """
        for var in self.vars:
            if var.adaptation is True:
                if len(var.shape) > 1:
                    torch.nn.init.kaiming_normal_(var)
                else:
                    torch.nn.init.zeros_(var)
         
    def reset_vars_meta_test(self):
        """
        Reset all adaptation parameters to random values. Bias terms are set to zero and other terms to default values of kaiming_normal_
        :return:
        """
        for var in self.vars:
            if 'linear' in var.id:
                if len(var.shape) > 1:
                    torch.nn.init.kaiming_normal_(var)
                else:
                    torch.nn.init.zeros_(var)
            
   
    '''
    def lstm_cell(self, x, states, w_ih, w_hh, b_ih, b_hh):
       """
       Single LSTM cell operation.
       """
       h_prev, c_prev = states

       # Compute gate activations
       gates = (
           torch.mm(x, w_ih.t()) + b_ih +
           torch.mm(h_prev, w_hh.t()) + b_hh
       )
       i, f, g, o = gates.chunk(4, dim=1)  # Split into 4 gate activations

       # Gate activations
       i = torch.sigmoid(i)  # Input gate
       f = torch.sigmoid(f)  # Forget gate
       g = torch.tanh(g)     # Cell candidate
       o = torch.sigmoid(o)  # Output gate

       # Update cell and hidden states
       c_next = f * c_prev + i * g  # Update cell state
       h_next = o * torch.tanh(c_next)  # Update hidden state

       return h_next, c_next  
   
    
    def forward(self, x, vars=None, config=None, sparsity_log=False, rep=False):

        x = x.float()
        if vars is None:
            vars = self.vars

        if config is None:
            config = self.config

        idx = 0

        for layer_counter, info_dict in enumerate(config):
            name = info_dict["name"]

            if name == 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                x = F.conv2d(x, w, b, stride=info_dict['config']['stride'], padding=info_dict['config']['padding'])
                idx += 2
                
            elif name == 'conv1d':
                w, b = vars[idx], vars[idx + 1]
                x = F.conv1d(x, w, b, stride=info_dict['config']['stride'], padding=info_dict['config']['padding'])
                idx += 2

            elif name == 'linear':            
                w, b = vars[idx], vars[idx + 1]

                x = F.linear(x, w, b)
                
                idx += 2
                
            elif name == 'lstm':          
                
                hidden_size = info_dict['config']['hidden_size']
                input_size = info_dict['config']['input_size']
                num_layers = info_dict['config']['num_layers']

                batch_size = x.size(0)
                h = [torch.zeros(batch_size, hidden_size, device=x.device, requires_grad=True) for _ in range(num_layers)]
                c = [torch.zeros(batch_size, hidden_size, device=x.device, requires_grad=True) for _ in range(num_layers)]

                for layer in range(num_layers):
                    w_ih = vars[idx]
                    w_hh = vars[idx + 1]
                    b_ih = vars[idx + 2]
                    b_hh = vars[idx + 3]
                    idx += 4

                    seq_len = x.size(1)
                    outputs = []
                    for t in range(seq_len):
                        x_t = x[:, t, :]
                        h[layer], c[layer] = self.lstm_cell(x_t, (h[layer], c[layer]), w_ih, w_hh, b_ih, b_hh)
                        outputs.append(h[layer])

                    x = torch.stack(outputs, dim=1)
                    x = F.layer_norm(x, x.size()[1:])  # Normalize outputs


            elif name == 'flatten':
                x = x.view(x.size(0), -1)

            elif name == 'maxPool1d':
                    x = F.max_pool1d(x,info_dict['config']['kernel'], info_dict['config']['stride'])


            elif name == 'dropout':
                    x = F.dropout(x,info_dict['config']['p'])

            elif name == 'rotate':
                # pass
                x = F.linear(x, self.rotate)
                x = F.linear(x, self.rotate_inverse)

            elif name == 'reshape':
                continue

            elif name == 'rep':
                if rep:
                    return x

            elif name == 'relu':
                x = F.relu(x)
                
                
            elif name == 'batchNorm1d':
                num_features = info_dict['config']['num_features']
                
                gamma = vars[idx]  # Scale parameter (shape: [num_features])
                beta = vars[idx + 1]  # Shift parameter (shape: [num_features])
                idx += 2
                
                eps = 1e-5  # Small constant for numerical stability
            
                # Determine correct dimension for mean/variance computation
                if x.dim() == 2:  # Shape: (batch_size, num_features)
                    mean = x.mean(dim=0, keepdim=True)  
                    var = x.var(dim=0, unbiased=False, keepdim=True)
                elif x.dim() == 3:  # Shape: (batch_size, num_features, seq_len)
                    mean = x.mean(dim=(0, 2), keepdim=True)  # Shape: (1, num_features, 1)
                    var = x.var(dim=(0, 2), unbiased=False, keepdim=True)  # Shape: (1, num_features, 1)
                else:
                    raise ValueError(f"Unexpected input shape {x.shape} for BatchNorm1d")
            
                # Ensure running mean/var have the correct shape
                if not hasattr(self, "running_mean"):
                    self.running_mean = torch.zeros((1, num_features, 1), device=x.device)
                    self.running_var = torch.ones((1, num_features, 1), device=x.device)
            
                if self.training:
                    # Ensure running stats have the correct shape before updating
                    self.running_mean = self.running_mean.to(mean.shape)  # Match mean shape
                    self.running_var = self.running_var.to(var.shape)  # Match var shape
            
                    # Update running statistics with momentum
                    momentum = 0.1
                    self.running_mean = momentum * mean + (1 - momentum) * self.running_mean
                    self.running_var = momentum * var + (1 - momentum) * self.running_var
                else:
                    mean = self.running_mean
                    var = self.running_var
            
                # Normalize x
                x = (x - mean) / torch.sqrt(var + eps)
            
                # Ensure gamma and beta are broadcastable
                gamma = gamma.view(1, -1, 1)  # Shape: (1, num_features, 1)
                beta = beta.view(1, -1, 1)  # Shape: (1, num_features, 1)
            
                # Scale and shift
                x = gamma * x + beta
            else:
                raise NotImplementedError
  
        assert idx == len(vars)
        

        return x

    def update_weights(self, vars):

        for old, new in zip(self.vars, vars):
            #old.data = new.data
            old.data = copy.deepcopy(new.data)


    def get_adaptation_parameters(self, vars=None):
        """
        :return: adaptation parameters i.e. parameters changed in the inner loop
        """
        if vars is None:
            vars = self.vars
            
        return list(filter(lambda x: x.adaptation, list(vars)))
    
    def get_adaptation_parameters_meta_test(self, vars=None):
        """
        :return: adaptation parameters i.e. parameters changed in the inner loop
        """
        if vars is None:
            vars = self.vars
            
        return [vars[-4], vars[-3], vars[-2],vars[-1]]
        
        return list(filter(lambda x: 'linear' in x.idx, list(vars)))    
    
    def get_forward_meta_parameters(self):
        """
        :return: adaptation parameters i.e. parameters changed in the inner loop
        """
        return list(filter(lambda x: x.meta, list(self.vars)))
    
    def clip_gradients(self):
      torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5)

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """

        return self.vars



    def reset_vars(self):
        """
        Reset all adaptation parameters to random values.
        Bias terms are set to zero, and other terms to default values of kaiming_normal_.
        LSTM parameters are initialized with Xavier uniform for weight matrices and zero for biases.
        """
        for var in self.vars:
            if var.adaptation is True:
                if hasattr(var, 'id') and 'lstm' in var.id:  # Check if it's an LSTM parameter
                    if 'weight' in var.id:
                        torch.nn.init.xavier_uniform_(var)
                    elif 'bias' in var.id:
                        torch.nn.init.zeros_(var)
                else:
                    if len(var.shape) > 1:
                        torch.nn.init.kaiming_normal_(var)
                    else:
                        torch.nn.init.zeros_(var)

    def reset_vars_meta_test(self):
        """
        Reset all adaptation parameters to random values.
        Bias terms are set to zero, and other terms to default values of kaiming_normal_.
        LSTM parameters are initialized with Xavier uniform for weight matrices and zero for biases.
        """
        for var in self.vars:
            if 'linear' in var.id:
                if len(var.shape) > 1:
                    torch.nn.init.kaiming_normal_(var)
                else:
                    torch.nn.init.zeros_(var)
            elif 'lstm' in var.id:
                if 'weight' in var.id:
                    torch.nn.init.xavier_uniform_(var)
                elif 'bias' in var.id:
                    torch.nn.init.zeros_(var)
                    
                   