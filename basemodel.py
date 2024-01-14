import torch
import torch.nn as nn
from conf import *

class HeatPINN(nn.Module):
    '''
    Base class for PINN to solve heat equation in a 2-dimensional space
    '''
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha
        self.hidden = nn.ModuleList()

        self.hidden.append(nn.Linear(2 + 1, 32))
        for _ in range(6):
            self.hidden.append(nn.Linear(32, 32))
        self.output = nn.Linear(32, 1)

       
    def forward(self, x):
        for fc in self.hidden:
            x = torch.tanh(fc(x))
        return self.output(x)
    

    # Helper function to calculate gradients
    def pde_loss(self, num_points: int):
        inputs = self.rand_domain_points(num_points).to(device)
        inputs.requires_grad = True

        u_pred = self(inputs)
        u_grad = torch.autograd.grad(u_pred, inputs, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        
        u_x = u_grad[:, 0]
        u_y = u_grad[:, 1]
        u_t = u_grad[:, 2]

        u_xx = torch.autograd.grad(u_x, inputs, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0]
        u_yy = torch.autograd.grad(u_y, inputs, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1]
        
        val = u_t - self.alpha * (u_xx + u_yy)

        return torch.mean(val ** 2)
    
    
    def init_loss(self, num_points: int):
        raise NotImplementedError()

    
    def boundary_loss(self, num_points: int):
        raise NotImplementedError()
    
    @staticmethod
    def rand_domain_points(num_points):
        raise NotImplementedError()

