import torch
import torch.optim as optim
import numpy as np

from model import HeatPINN2
from conf import *

import os

if __name__ == '__main__':
    model = HeatPINN2(alpha=1, u0=0, u1=1, u2=0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 20000
    BATCH_SIZE = 10000

    min_loss = float('inf')

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        loss = model.pde_loss(BATCH_SIZE) + model.init_loss(BATCH_SIZE) + model.boundary_loss(BATCH_SIZE)
        loss.backward()
        optimizer.step()
        if loss.item() < min_loss:
            min_loss = loss.item()
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), 'model.pth'))
        if epoch % 500 == 499:
            print(f'Epoch {epoch + 1}/{EPOCHS}: Loss {loss.item():.4f}')