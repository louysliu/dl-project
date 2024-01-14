import torch
import torch.optim as optim
import numpy as np

from model import HeatPINN1
from conf import *

import os

if __name__ == '__main__':
    model = HeatPINN1(alpha=1.0, u0=0, u1=1.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    EPOCHS = 20000
    BATCH_SIZE = 10000

    min_loss = float('inf')
    best_state_dict = None
    losses = []

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        loss = model.pde_loss(BATCH_SIZE) + model.init_loss(BATCH_SIZE) + model.boundary_loss(BATCH_SIZE)
        loss.backward()
        optimizer.step()
        ls = loss.item()
        losses.append(ls)
        if ls < min_loss:
            min_loss = ls
            best_state_dict = model.state_dict().copy()
        if epoch % 500 == 499:
            print(f'Epoch {epoch + 1}/{EPOCHS}: Loss {loss.item():.5f}')
    
    model.load_state_dict(best_state_dict)
    torch.save(model, os.path.join(os.path.dirname(__file__), 'model.pth'))

    # draw loss curve and save as png
    # log scale
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(os.path.dirname(__file__), 'loss.png'))
    plt.close()
