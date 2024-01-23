#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from model import diffusion_model
import torch
from grid import volume_grid
import torch.nn as nn
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


def mtl_training_1(nn_models, hjm_model, optimizer_configs, grid, strike_price, Q_ints, Q_ints_for_training, log = False):
    """ Learning phase of the MTL: EMAG scheme
    
    Args:
        nn_models (1D array): list of neural network per exercise date
        optimizer_configs (dict): optimizer configurations
        grid (class): swing volume grid
        strike_price (float): fixed strike price
        Q_ints (1D array): list of attainable cumulative volume per date
        Q_ints_for_training (1D array): list of tasks to train (except trivial tasks) per date
    """
    
    n_dates = len(grid.ex_dates)
    last_shared_layer_index = nn_models[1].n_shared_layer * 3 - 3 # index of the last shared layers (shared module)
    optimizers = [torch.optim.Adam(model.parameters(), lr = optimizer_configs['lr_1']) for model in nn_models]
    optimizers_W = [0] * (n_dates - 1)  
    comp = 0
    iters = np.full(n_dates - 1, 0)
    weights = [0] * (n_dates - 1)
    L_ema = [0] * (n_dates - 1) # storage for EMA losses
    my_loss = {'price' : [], 'dl': []}
    norm_Q = grid.Q_max - grid.Q_min
            
    for epoch in range(optimizer_configs['n_iterations']):
        
        for batch in range(optimizer_configs['n_batches']):
            X_train = hjm_model.Simulate_X(optimizer_configs['batch_size'])
            psi_train = hjm_model.Compute_Spot_Price_From_X(X_train) - strike_price
            next_values = torch.zeros((len(Q_ints[n_dates - 1]), optimizer_configs['batch_size'], 1), device = device)
            unit_tensor = torch.ones((optimizer_configs['batch_size'], 1), device = device)
                            
            for n in range(n_dates - 1, -1, -1): 
                flag_lb = False
                
                if n > 0 and n < n_dates - 1:
                    # transfer learning: start with just trained shared representation
                    nn_models[n].shared_layers.load_state_dict(nn_models[n + 1].shared_layers.state_dict())

                current_values = torch.zeros((len(Q_ints[n]), optimizer_configs['batch_size'], 1), device = device)
                Q = Q_ints_for_training[n]
                
                # compute loss for trivial tasks
                if not Q_ints[n][0] in Q:
                    flag_lb = True
                    q_plus = grid.Compute_q_plus(Q_ints[n][0], n)
                    current_values[0] = q_plus * psi_train[n][:, np.newaxis] + next_values[0]
                    
                if not Q_ints[n][-1] in Q:
                    q_moins = grid.Compute_q_moins(Q_ints[n][-1], n)
                    current_values[len(Q_ints[n]) - 1] = q_moins * psi_train[n][:, np.newaxis]
                    
                    if n < n_dates - 1:
                        current_values[len(Q_ints[n]) - 1] += next_values[len(Q_ints[n + 1]) - 1]
                
                # Compute loss per task
                inputs = torch.cat((X_train[n], psi_train[n][:, np.newaxis]), 1)
                xi_per_task = nn_models[n](inputs)
                n_tasks = len(xi_per_task[0]) if n > 0 else 1
                loss_per_task = torch.zeros(n_tasks, device = device)
                                
                for j in range(n_tasks):
                    xi = xi_per_task[:, j, :] if n > 0 else xi_per_task
                    xi_bg = (xi >= 0.5).detach().float()
                    q_plus = grid.Compute_q_plus(Q[j], n)
                    q_moins = grid.Compute_q_moins(Q[j], n)
                    cf_plus = q_plus * psi_train[n][:, np.newaxis]
                    cf_moins = q_moins * psi_train[n][:, np.newaxis]

                    if n < n_dates - 1:
                        index_plus = (Q[j] + q_plus - Q_ints[n + 1][0]).long()
                        index_moins = (Q[j] + q_moins - Q_ints[n + 1][0]).long()
                        cf_plus += next_values[index_plus]
                        cf_moins += next_values[index_moins]
                                                
                    loss_per_task[j] = -torch.mean(cf_plus * xi + cf_moins * (1.0 - xi))
                    start_index = j + 1 if flag_lb else j # whether or not first trivial loss has already been computed
                    current_values[start_index] = cf_plus * xi_bg + cf_moins * (1.0 - xi_bg)
                    
                next_values = copy.deepcopy(current_values) # update next values
                optimizers[n].zero_grad()
                
                # optimization
                if n == 0:
                    global_loss = loss_per_task[0]
                    global_loss.backward()
                    optimizers[0].step()
                    
                    #if log:
                        #for i in range(1, n_ex_dates):
                            #nn_models[i].sample_size = 50000
                            
                        #result = valuation(nn_models, grid, strike_price, 10, 50000, Q_ints, Q_ints_for_training)
                        #price = result["swing_price"]
                        #dl = 1.96 * round((result["var_price"] / 500000) ** 0.5 , 2)
                        #my_loss['price'].append(price)
                        #my_loss['dl'].append(dl)
                        
                        #for i in range(1, n_ex_dates):
                            #nn_models[i].sample_size = optimizer_configs['batch_size']
                        
                else:
                    if iters[n - 1] == 0:
                        weights[n - 1] = torch.ones_like(loss_per_task)
                        weights[n - 1] = torch.nn.Parameter(weights[n - 1])
                        optimizers_W[n - 1] = torch.optim.Adam([weights[n - 1]], lr = optimizer_configs['lr_2'])
                        L_ema[n - 1] = loss_per_task.detach().clone()
                    
                    weighted_loss = weights[n - 1] @ loss_per_task
                    
                    if iters[n - 1] == 0:
                        weighted_loss.backward()
                        optimizers[n].step()
                        
                    else:
                        weighted_loss.backward(retain_graph = True)
                        gw = []
                
                        for j in range(n_tasks):
                            dl = torch.autograd.grad(weights[n - 1][j] * loss_per_task[j], nn_models[n].shared_layers[last_shared_layer_index].parameters(), 
                                                 retain_graph = True, create_graph = True)[0]
                            gw.append(torch.norm(dl))
                        
                        gw = torch.stack(gw)
                        gw_avg = gw.mean().detach()
                        Bt = nn.Sigmoid()((loss_per_task - L_ema[n - 1]) * optimizer_configs['tmp']).detach()
                        rt = Bt / Bt.mean()
                        loss_w = torch.abs(gw - gw_avg * (rt ** optimizer_configs['b'])).sum()
                        optimizers_W[n - 1].zero_grad()
                        loss_w.backward() 
                        optimizers[n].step()
                        optimizers_W[n - 1].step()
                        weights[n - 1] = torch.nn.Parameter(((weights[n - 1] / weights[n - 1].sum()) * n_tasks).detach())
                        optimizers_W[n - 1] = torch.optim.Adam([weights[n - 1]], lr = optimizer_configs['lr_2'])
                        L_ema[n - 1] = optimizer_configs['beta'] * L_ema[n - 1] + (1.0 - optimizer_configs['beta']) * loss_per_task.detach().clone()
                        
                    iters[n - 1] = 1
                    
    return nn_models, my_loss


# In[ ]:


def Weight_Strat(strat, number_of_tasks):
    if strat == "equal":
        return torch.ones(number_of_tasks, device = device)
    
    if strat == "uniform":
        return torch.rand(number_of_tasks).to(device)


# In[1]:


def mtl_training_2(nn_models, hjm_model, optimizer_configs, grid, strike_price, Q_ints, Q_ints_for_training, log = False):
    """ Learning phase of the MTL: basics weights strategy
    
    Args:
        nn_models (1D array): list of neural network per exercise date
        optimizer_configs (dict): optimizer configurations
        grid (class): swing volume grid
        strike_price (float): fixed strike price
        Q_ints (1D array): list of attainable cumulative volume per date
        Q_ints_for_training (1D array): list of tasks to train (except trivial tasks) per date
    """
    
    n_dates = len(grid.ex_dates)
    last_shared_layer_index = nn_models[1].n_shared_layer * 3 - 3
    optimizers = [torch.optim.Adam(model.parameters(), lr = optimizer_configs['lr']) for model in nn_models]
    comp = 0
    my_loss = {'price' : [], 'dl': []}
    norm_Q = grid.Q_max - grid.Q_min
            
    for epoch in range(optimizer_configs['n_iterations']):
        
        for batch in range(optimizer_configs['n_batches']):
            X_train = hjm_model.Simulate_X(optimizer_configs['batch_size'])
            psi_train = hjm_model.Compute_Spot_Price_From_X(X_train) - strike_price
            next_values = torch.zeros((len(Q_ints[n_dates - 1]), optimizer_configs['batch_size'], 1), device = device)
            unit_tensor = torch.ones((optimizer_configs['batch_size'], 1), device = device)
                            
            for n in range(n_dates - 1, -1, -1): 
                flag_lb = False
                
                if n > 0 and n < n_dates - 1:
                    nn_models[n].shared_layers.load_state_dict(nn_models[n + 1].shared_layers.state_dict())

                current_values = torch.zeros((len(Q_ints[n]), optimizer_configs['batch_size'], 1), device = device)
                Q = Q_ints_for_training[n]
                
                if not Q_ints[n][0] in Q:
                    flag_lb = True
                    q_plus = grid.Compute_q_plus(Q_ints[n][0], n)
                    current_values[0] = q_plus * psi_train[n][:, np.newaxis] + next_values[0]
                    
                if not Q_ints[n][-1] in Q:
                    q_moins = grid.Compute_q_moins(Q_ints[n][-1], n)
                    current_values[len(Q_ints[n]) - 1] = q_moins * psi_train[n][:, np.newaxis]
                    
                    if n < n_dates - 1:
                        current_values[len(Q_ints[n]) - 1] += next_values[len(Q_ints[n + 1]) - 1]
                
                inputs = torch.cat((X_train[n], psi_train[n][:, np.newaxis]), 1)
                xi_per_task = nn_models[n](inputs)
                n_tasks = len(xi_per_task[0]) if n > 0 else 1
                loss_per_task = torch.zeros(n_tasks, device = device)
                                
                for j in range(n_tasks):
                    xi = xi_per_task[:, j, :] if n > 0 else xi_per_task
                    xi_bg = (xi >= 0.5).detach().float()
                    q_plus = grid.Compute_q_plus(Q[j], n)
                    q_moins = grid.Compute_q_moins(Q[j], n)
                    cf_plus = q_plus * psi_train[n][:, np.newaxis]
                    cf_moins = q_moins * psi_train[n][:, np.newaxis]

                    if n < n_dates - 1:
                        index_plus = (Q[j] + q_plus - Q_ints[n + 1][0]).long()
                        index_moins = (Q[j] + q_moins - Q_ints[n + 1][0]).long()
                        cf_plus += next_values[index_plus]
                        cf_moins += next_values[index_moins]
                                                
                    loss_per_task[j] = -torch.mean(cf_plus * xi + cf_moins * (1.0 - xi))
                    start_index = j + 1 if flag_lb else j
                    current_values[start_index] = cf_plus * xi_bg + cf_moins * (1.0 - xi_bg)
                    
                next_values = copy.deepcopy(current_values)
                optimizers[n].zero_grad()
                weights = Weight_Strat(optimizer_configs['weights_strat'], n_tasks)
                weights /= weights.sum()
                weighted_loss = weights @ loss_per_task
                weighted_loss.backward() 
                optimizers[n].step()
                
                #if log and n == 0:
                    #for i in range(1, n_ex_dates):
                        #nn_models[i].sample_size = 50000
                            
                    #result = valuation(nn_models, grid, strike_price, 10, 50000, Q_ints, Q_ints_for_training)
                    #price = result["swing_price"]
                    #dl = 1.96 * round((result["var_price"] / 500000) ** 0.5 , 2)
                    #my_loss['price'].append(price)
                    #my_loss['dl'].append(dl)
                        
                    #for i in range(1, n_ex_dates):
                        #nn_models[i].sample_size = optimizer_configs['batch_size']
            
    return nn_models, my_loss


# In[ ]:


def valuation(mtl_trained_model, hjm_model, grid, strike_price, n_pack, pack_size, Q_ints, Q_ints_for_training):
    """ Evaluation phase of the MTL
    
    Args:
        mtl_trained_model (module): a trained MTL network
        grid (class): swing volume grid
        strike_price (float): fixed strike price
        n_pack (int): number of packs to be generated for a sequential valuation
        pack_size (int): size of each pack
        Q_ints (1D array): list of attainable cumulative volume per date
        Q_ints_for_training (1D array): list of tasks to train (except trivial tasks) per date
    """
    
    swing_price = 0.0
    var_price = 0.0
    n_dates = len(grid.ex_dates)
    M = float(n_pack * pack_size) ** 0.5
    
    with torch.no_grad():
        for i in range(n_pack):
            X_test = hjm_model.Simulate_X(pack_size)
            psi_test = hjm_model.Compute_Spot_Price_From_X(X_test) - strike_price
            Q = torch.zeros_like(psi_test[0][:, np.newaxis])
            zeros_tens = torch.zeros(pack_size, 1, 1).to(device)
            unit_tens = torch.ones(pack_size, 1, 1).to(device)
            running_cf = torch.zeros((pack_size, 1)).to(device)
            
            for n in range(n_dates):
                inputs = torch.cat((X_test[n], psi_test[n][:, np.newaxis]), 1)
                xi_per_task = mtl_trained_model[n](inputs)
                
                if n == 0:
                    xi = xi_per_task
                    
                else:
                    indexes = (Q - Q_ints[n][0]).long().reshape(pack_size) # a revoir
                    
                    if not Q_ints[n][0] in Q_ints_for_training[n]:
                        xi_per_task = torch.cat((unit_tens, xi_per_task), 1)
                        
                    if not Q_ints[n][-1] in Q_ints_for_training[n]:
                        xi_per_task = torch.cat((xi_per_task, zeros_tens), 1)
                    
                    xi = xi_per_task[range(len(indexes)), indexes]
                    
                xi_bang_bang = (xi >= 0.5).detach().float()
                q = grid.Compute_Constrained_Control(xi_bang_bang, Q, n)
                running_cf += q * psi_test[n][:, np.newaxis]
                Q += q.detach()
                
            swing_price += torch.mean(running_cf).item()
            var_price += torch.sum(torch.pow(running_cf / M, 2.0)).item()
                
        swing_price /= n_pack
        var_price -= swing_price ** 2
        
    return {"swing_price" : round(swing_price, 2), "var_price" : round(var_price, 2)}          

