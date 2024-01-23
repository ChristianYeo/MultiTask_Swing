#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from abc import abstractmethod
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


class Volume_Grid:
    
    """ Swing volume grid (constraints space) management
    
    Args:
        Q_min (float): global constraint min
        Q_max (float): global constraint max
        q_min (1D tensor): local constraint min
        q_max (1D tensor): local constraint max
        ex_dates (numpy.array): exercise dates
    """
    
    def __init__(self, Q_min, Q_max, q_min, q_max, ex_dates):
        
        self.Q_min = Q_min 
        self.Q_max = Q_max
        self.ex_dates = ex_dates
        self.q_min = q_min
        self.q_max = q_max
        
        zeros_tensor = torch.zeros(len(q_min))
        unit_tensor = torch.ones(len(q_min))
        n_ex_dates = len(ex_dates)
            
        # Upper Bound Volume Grid
        self.upper_bound = torch.zeros(n_ex_dates + 1)
        self.upper_bound[0] = 0.0
        self.upper_bound[n_ex_dates] = self.Q_max
        self.upper_bound[1 : n_ex_dates] = torch.minimum(torch.cumsum(self.q_max, 0)[:-1], self.Q_max * unit_tensor[:-1])
        
        # Lower Bound Volume Grid
        self.lower_bound = torch.zeros(n_ex_dates + 1)
        self.lower_bound[n_ex_dates] = self.Q_min
        self.lower_bound[0] = 0.0        
        self.lower_bound[1 : n_ex_dates] = torch.flip(torch.maximum(zeros_tensor[:-1], self.Q_min - torch.cumsum(torch.flip(self.q_max, (0,)) , 0)[:-1]), (0,))
        
        
    def discretization_cum_vol(self):
        """Compute integer attainable cumulative volumes at each exercise date"""
        
        result = [torch.tensor([0])] + [torch.arange(int(self.lower_bound[i]), int(self.upper_bound[i]) + 1) for i in range(1, len(self.ex_dates))]
        result_for_training = [torch.tensor([0])]
        
        for i in range(1, len(self.ex_dates)):
            x = result[i]
            if not self.q_min[0] in x:
                x = x[1:]
                
            if self.Q_max in result[i]:
                x = x[:-1]
                
            result_for_training.append(x)
                
        return result, result_for_training
    
    def Compute_q_plus(self, Q, n):
        """Compute maximum consumption at an exercise date given a cumulative consumption"""
        
        return torch.minimum(self.upper_bound[n + 1] - Q, self.q_max[n])
    
    def Compute_q_moins(self, Q, n):
        """Compute maximum consumption at an exercise date given a cumulative consumption"""
        
        return torch.maximum(self.lower_bound[n + 1] - Q, self.q_min[n])
    
    def Compute_Constrained_Control(self, xi, Q, n):
        """Compute consumption at an exercise date given a decisions"""
        
        A1 = self.Compute_q_moins(Q, n)
        A2 = self.Compute_q_plus(Q, n)
        
        return A1 + (A2 - A1) * xi

