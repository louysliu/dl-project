import torch
import torch.optim as optim
import numpy as np

from model import HeatPINN1
from conf import *

import os

if __name__ == '__main__':
    model = HeatPINN1(alpha=1.0, u0=0, u1=1.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 10000
    BATCH_SIZE = 10000

    min_loss = float('inf')
    best_state_dict = None

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        loss = model.pde_loss(BATCH_SIZE) + model.init_loss(BATCH_SIZE) + model.boundary_loss(BATCH_SIZE)
        loss.backward()
        optimizer.step()
        if loss.item() < min_loss:
            min_loss = loss.item()
            best_state_dict = model.state_dict().copy()
        if epoch % 500 == 499:
            print(f'Epoch {epoch + 1}/{EPOCHS}: Loss {loss.item():.5f}')
    
    model.load_state_dict(best_state_dict)
    torch.save(model, os.path.join(os.path.dirname(__file__), 'model.pth'))