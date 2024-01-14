import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basemodel import HeatPINN
from conf import *
import torch


class HeatPINN2(HeatPINN):
    ''' 
    PINN to solve heat equation in a 2-dimensional space \\
    u_t = alpha * (u_xx + u_yy), t > 0, 0 < x < 1, 0 < y < 1 \\
    u(x, y, 0) = u0 \\
    u(x, 1, t) = u1 \\
    u(x, 0, t) = u(1, y, t) = u(0, y, t) = u2
    '''
    def __init__(self, alpha: float, u0: float, u1: float, u2: float):
        super().__init__(alpha)
        self.u0 = u0
        self.u1 = u1
        self.u2 = u2
    
    def init_loss(self, num_points: int):
        # initial condition u(x, y, 0) = u0
        inputs = self.rand_init_points(num_points).to(device)
        u_pred = self(inputs)
        return torch.mean((u_pred - self.u0) ** 2)
    
    def boundary_loss(self, num_points: int):
        # Boundary condition
        num_top = num_points // 4
        num_other = num_points - num_top
        return (num_top * self.top_boundary_loss(num_top) + num_other * self.other_boundary_loss(num_other)) / num_points
    
    def top_boundary_loss(self, num_points: int):
        # u(x, 1, t) = u1
        inputs = self.rand_top_boundary_points(num_points).to(device)
        u_pred = self(inputs)
        return torch.mean((u_pred - self.u1) ** 2)
    
    def other_boundary_loss(self, num_points: int):
        # u(x, 0, t) = u(1, y, t) = u(0, y, t) = u2
        inputs = self.rand_other_boundary_points(num_points).to(device)
        u_pred = self(inputs)
        return torch.mean((u_pred - self.u2) ** 2)
        
    @staticmethod
    def rand_domain_points(num_points: int):
        points = torch.rand((num_points, 2)) # [0, 1) * [0, 1)
        clocks = torch.rand((num_points, 1)) * 30
        inputs = torch.cat((points, clocks), dim=1)
        return inputs
    
    @staticmethod
    def rand_init_points(num_points: int):
        points = torch.rand((num_points, 2)) # [0, 1) * [0, 1)
        inputs = torch.cat((points, torch.zeros((num_points, 1))), dim=1)
        return inputs
    
    @staticmethod
    def rand_top_boundary_points(num_points: int):
        clocks = torch.rand((num_points, 1)) * 30
        xs = torch.rand((num_points, 1)) # [0, 1)
        ys = torch.ones((num_points, 1)) # [0, 1)
        inputs = torch.cat((xs, ys, clocks), dim=1)
        return inputs
    
    @staticmethod
    def rand_other_boundary_points(num_points: int):
        num_side = num_points // 3
        num_bottom = num_points - num_side * 2
        
        # bottom
        xs = torch.rand((num_bottom, 1))
        ys = torch.zeros((num_bottom, 1))
        clocks = torch.rand((num_bottom, 1)) * 30
        bottoms = torch.cat((xs, ys, clocks), dim=1)

        # left
        xs = torch.zeros((num_side, 1))
        ys = torch.rand((num_side, 1))
        clocks = torch.rand((num_side, 1)) * 30
        lefts = torch.cat((xs, ys, clocks), dim=1)

        # right
        xs = torch.ones((num_side, 1))
        ys = torch.rand((num_side, 1))
        clocks = torch.rand((num_side, 1)) * 30
        rights = torch.cat((xs, ys, clocks), dim=1)

        inputs = torch.cat((bottoms, lefts, rights), dim=0)
        return inputs

