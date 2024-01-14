import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basemodel import HeatPINN
from conf import *
import torch
import torch.nn as nn


class HeatPrecise1(nn.Module):
    def __init__(self, alpha: float, u0: float, u1: float):
        super().__init__()
        self.alpha = alpha
        self.u0 = u0
        self.u1 = u1

    def forward(self, input):
        x = input[:, 0]
        y = input[:, 1]
        t = input[:, 2]

        tmp = 2 * torch.sqrt(self.alpha * t)
        return (self.u1 - self.u0) * torch.erfc((x**2+y**2) / tmp) + self.u0


class HeatPINN1(HeatPINN):
    ''' 
    PINN to solve heat equation in a n-dimensional space \\
    u_t = alpha * (u_xx + u_yy), t >= 0, |x| <= 1, |y| <= 1 \\
    u(x, y, 0) = u0 \\
    u(0, 0, t) = u1
    '''
    def __init__(self, alpha: float, u0: float, u1: float):
        super().__init__(alpha)
        self.u0 = u0
        self.u1 = u1
    
    
    def init_loss(self, num_points: int):
        # initial condition u(x, y, 0) = u0
        inputs = self.rand_init_points(num_points).to(device)
        u_pred = self(inputs)
        return torch.mean((u_pred - self.u0) ** 2)

    
    def boundary_loss(self, num_points: int):
        # Boundary condition u(0, 0, t) = u1
        inputs = self.rand_boundary_points(num_points).to(device)
        u_pred = self(inputs)
        return torch.mean((u_pred - self.u1) ** 2)
        
    @staticmethod
    def rand_domain_points(num_points: int):
        points = torch.rand((num_points, 2)) * 20 - 10
        clocks = torch.rand((num_points, 1)) * 30
        inputs = torch.cat((points, clocks), dim=1)
        return inputs
    
    @staticmethod
    def rand_init_points(num_points: int):
        points = torch.rand((num_points, 2)) * 20 - 10
        inputs = torch.cat((points, torch.zeros((num_points, 1))), dim=1)
        return inputs
    
    @staticmethod
    def rand_boundary_points(num_points: int):
        clocks = torch.rand((num_points, 1)) * 30 # [0, 60)
        inputs = torch.cat((torch.zeros((num_points, 2)), clocks), dim=1)
        return inputs
    