#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


class MTL_nn(nn.Module):
    """ Multitask neural network architecture
    
    Args:
        n_features (int): inputs dimension
        hidden_shared_layers (1D array): number of neurons per hidden layer
        Q_ints_training (1D tensor): list of cumulative volumes for the training: except trivial tasks
        Q_min (float): swing contract minimal global contraint
        Q_norm (float): a volume normalization coefficient = Q_max - Q_min
        sample_size (int): sample size
    """
    
    def __init__(self, n_features, hidden_shared_layers, Q_ints_training, Q_min, norm_Q, sample_size):
        super().__init__()

        self.n_shared_layer = len(hidden_shared_layers) # number of hidden layers in the shared module
        self.Q_ints_training = Q_ints_training
        self.Q_min = Q_min
        self.norm_Q = norm_Q
        self.sample_size = sample_size
        
        n_tasks = len(Q_ints_training)
        last_layer_index = self.n_shared_layer - 1
        
        # shared modules
        modules = []
        prev_dim = n_features
        
        for i in range(0, self.n_shared_layer - 1):
            modules.append(nn.Linear(prev_dim, hidden_shared_layers[i]))
            modules.append(nn.BatchNorm1d(hidden_shared_layers[i]))
            modules.append(nn.ReLU())
            prev_dim = hidden_shared_layers[i]
            
        modules.append(nn.Linear(prev_dim, hidden_shared_layers[last_layer_index]))
        prev_dim = hidden_shared_layers[last_layer_index]
        self.shared_layers = nn.Sequential(*modules).to(device)
            
        # task specific modules
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(prev_dim, 2), nn.BatchNorm1d(2)) for _ in range(n_tasks)]).to(device)
        
    def forward(self, x):
        y_shared_modules = self.shared_layers(x)
        y_per_task = []
        unit_tensor = torch.ones((self.sample_size, 1), device = device)

        for i, head in enumerate(self.heads):
            inputs_for_decision = head(y_shared_modules) # output \chi_k
            MQ = torch.cat((((self.Q_ints_training[i] - self.Q_min) / self.norm_Q) * unit_tensor, unit_tensor), 1)   
            y_per_task.append(nn.Sigmoid()(torch.sum(inputs_for_decision * MQ, 1))[:, np.newaxis]) # function f_k^i
            
        return torch.stack(y_per_task, 1)


# In[ ]:


def Build_Mtl_Architecture_For_Swing(n_features, hidden_shared_layers, Q_ints_training, Q_min, norm_Q, sample_size, n_unit_initial_nn):
    """ Build MTL neural net architecture for the pricing of swing contracts: one feedforward net for the first date and MTL nets for the remaining dates
    
    Args:
        n_features (int): inputs dimension
        hidden_shared_layers (1D array): number of neurons per hidden layer
        Q_ints_training (2D array): list of tasks per date, expect trivial tasks
        Q_min (float): minimum global constraints
        norm_Q (float): Q_max - Q_min
        sample_size (int): sample_size
        n_unit_initial_nn (int): number of units for the hidden layer of the initial feedforward neural network
    """

    nn_model = [nn.Sequential(nn.Linear(n_features, n_unit_initial_nn), nn.BatchNorm1d(n_unit_initial_nn), 
                              nn.ReLU(), nn.Linear(n_unit_initial_nn, 1), nn.Sigmoid()).to(device)]
    
    for i in range(1, len(Q_ints_training)):
        nn_model += [MTL_nn(n_features, hidden_shared_layers, Q_ints_training[i], Q_min, norm_Q, sample_size)]
        
    return nn_model

