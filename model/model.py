#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import datetime
import _pickle as cPickle



class StatisticsContainer(object):
    def __init__(self, statistics_path):
        self.statistics_path = statistics_path

        self.backup_statistics_path = '{}_{}.pickle'.format(
            os.path.splitext(self.statistics_path)[0], 
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        self.statistics_dict = {'Trainloss': [], 'Testloss': [], 'test_f1': [], 'BaseTrainloss': [], 'BaseTrain_f1': [], 'Testloss_NewClasses':[], 'newClasses_test_f1':[],'Testloss_AllClasses':[], 'allClasses_test_f1':[]}

    def append(self, iteration, statistics, data_type):
        statistics['iteration'] = iteration
        self.statistics_dict[data_type].append(statistics)
        
    def dump(self):
        cPickle.dump(self.statistics_dict, open(self.statistics_path, 'wb'))
        cPickle.dump(self.statistics_dict, open(self.backup_statistics_path, 'wb'))
        
    def load_state_dict(self, resume_iteration):
        self.statistics_dict = cPickle.load(open(self.statistics_path, 'rb'))

        resume_statistics_dict = {'Trainloss': [], 'Testloss': [], 'test_f1': [], 'BaseTrainloss': [], 'BaseTrain_f1': [], 'Testloss_NewClasses':[], 'newClasses_test_f1':[], 'Testloss_AllClasses':[], 'allClasses_test_f1':[]}
        
        for key in self.statistics_dict.keys():
            for statistics in self.statistics_dict[key]:
                if statistics['iteration'] <= resume_iteration:
                    resume_statistics_dict[key].append(statistics)
                
        self.statistics_dict = resume_statistics_dict


class ForgettingContainer(object):
    def __init__(self, statistics_path):
        self.statistics_path = statistics_path

        self.backup_statistics_path = '{}_{}.pickle'.format(
            os.path.splitext(self.statistics_path)[0], 
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        self.statistics_dict = {'ForgettingScore': []}

    def append(self, iteration, statistics, data_type):
        statistics['iteration'] = iteration
        self.statistics_dict[data_type].append(statistics)
        
    def dump(self):
        cPickle.dump(self.statistics_dict, open(self.statistics_path, 'wb'))
        cPickle.dump(self.statistics_dict, open(self.backup_statistics_path, 'wb'))
        
    def load_state_dict(self, resume_iteration):
        self.statistics_dict = cPickle.load(open(self.statistics_path, 'rb'))

        resume_statistics_dict = {'ForgettingScore': []}
        
        for key in self.statistics_dict.keys():
            for statistics in self.statistics_dict[key]:
                if statistics['iteration'] <= resume_iteration:
                    resume_statistics_dict[key].append(statistics)
                
        self.statistics_dict = resume_statistics_dict 

def init_layer(layer):

    if type(layer) == nn.LSTM:
        for name, param in layer.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    else:
        """Initialize a Linear or Convolutional layer. """
        nn.init.xavier_uniform_(layer.weight)
 
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)




class DeepConvLSTM(nn.Module):
    '''
    def __init__(self, n_classes, NB_SENSOR_CHANNELS, SLIDING_WINDOW_LENGTH, n_hidden=128, n_layers=1, n_filters=64, 
                filter_size=5, drop_prob=0.5, ):
        super(DeepConvLSTM, self).__init__()

        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.filter_size = filter_size
        self.NB_SENSOR_CHANNELS = NB_SENSOR_CHANNELS
        self.SLIDING_WINDOW_LENGTH = SLIDING_WINDOW_LENGTH
             
        self.conv1 = nn.Conv1d(self.NB_SENSOR_CHANNELS, n_filters, filter_size)
        self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size)
        self.conv3 = nn.Conv1d(n_filters, n_filters, filter_size)
        self.conv4 = nn.Conv1d(n_filters, n_filters, filter_size)
        
        self.lstm1  = nn.LSTM(n_filters, n_hidden, n_layers)
        self.lstm2  = nn.LSTM(n_hidden, n_hidden, n_layers)
        
        self.fc = nn.Linear(n_hidden, n_classes)

        self.dropout = nn.Dropout(drop_prob)

        self.init_weight()
        '''
    def __init__(self, n_classes, channels, data_size,cnn_layers, kernel, stride,out_linear,  n_hidden, n_layers):
         super(DeepConvLSTM, self).__init__()
  
         #maxpool_out_dim = (data_size[1] - ((kernel * cnn_layers) - (stride * cnn_layers))) // 2
         #flatten_out_dim = int(channels * maxpool_out_dim)
         
         self.n_hidden = n_hidden
         self.n_filters = channels
         self.n_classes = n_classes
         self.n_layers = n_layers
         self.NB_SENSOR_CHANNELS = data_size[0]
         
         self.conv1 = nn.Conv1d(in_channels=data_size[0], out_channels= channels, kernel_size=kernel, stride=stride)
         self.conv2 = nn.Conv1d(in_channels=channels, out_channels= channels, kernel_size=kernel, stride=stride)
         self.conv3 = nn.Conv1d(in_channels=channels, out_channels= channels, kernel_size=kernel, stride=stride)
         self.conv4 = nn.Conv1d(in_channels=channels, out_channels= channels, kernel_size=kernel, stride=stride)
         
         self.lstm1  = nn.LSTM(channels, n_hidden, n_layers)
         self.lstm2  = nn.LSTM(n_hidden, n_hidden, n_layers)
        
         self.fc = nn.Linear(n_hidden, n_classes)
 
         self.dropout = nn.Dropout(0.6)

         self.init_weight()

    def init_weight(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)
        init_layer(self.lstm1)
        init_layer(self.lstm2)
        init_layer(self.fc)        

    def forward(self, x, hidden, batch_size):
        print('FORWARD')
        print('x.shape',x.shape)
        #print('self.NB_SENSOR_CHANNELS',self.NB_SENSOR_CHANNELS)
        #print('self.SLIDING_WINDOW_LENGTH',self.SLIDING_WINDOW_LENGTH)
        #x = x.view(-1, self.NB_SENSOR_CHANNELS, self.SLIDING_WINDOW_LENGTH)
        print(x.shape)
       # x = x.permute(0,2,1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        print(x.shape)
        x = x.view(x.shape[-1], -1, self.n_filters)
        print(x.shape)

        print(np.shape(x), np.shape(hidden))
        x = self.dropout(x)
        x, hidden = self.lstm1(x, hidden)
        print(x.shape)

        x, hidden = self.lstm2(x, hidden)
        print(x.shape)

        print(np.shape(x))

        x = x.contiguous().view(-1, self.n_hidden)
        embeddings = x.contiguous().view(batch_size,-1,self.n_hidden)[:,-1,:]
        x = torch.sigmoid(self.fc(x))
        print(np.shape(x))
        temp = x.view(batch_size, -1, self.n_classes)
        print(np.shape(temp))
        out = x.view(batch_size, -1, self.n_classes)[:,-1,:]
        
        return out, hidden, embeddings
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (torch.cuda.is_available()):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
        return hidden

### cosine normalization in last layer to level the difference of the embeddings and biases between all classes (from LUCIR)  ----- idea can apply to prototypes of all formed classes? 
class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1) #for initializaiton of sigma

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2,dim=1), \
                F.normalize(self.weight, p=2, dim=1)) # experiment with adding bias?
        if self.sigma is not None:
            out = self.sigma * out
        return out