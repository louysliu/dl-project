import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from model import HeatPINN2
import os


if __name__ == '__main__':
    basedir = os.path.dirname(__file__)

    # Visualize the model
    model = HeatPINN2(alpha=1, u0=0, u1=1, u2=0)
    model.load_state_dict(torch.load(os.path.join(basedir,'model.pth')))
    
    # Generate a grid of x and y values
    x_values = np.linspace(0, 1, 400)
    y_values = np.linspace(0, 1, 400)
    x_grid, y_grid = np.meshgrid(x_values, y_values)

    # Convert to PyTorch tensors
    x_tensor = torch.tensor(x_grid, dtype=torch.float32)
    y_tensor = torch.tensor(y_grid, dtype=torch.float32)

    # Time range for visualization
    t_values = np.linspace(0, 30, 151)  # Change this based on your specific needs

    # Prepare plots for animation
    fig, ax = plt.subplots()

    t_tensor = torch.full_like(x_tensor, fill_value=0.0, dtype=torch.float32)
    combined_input = torch.stack([x_tensor.flatten(), y_tensor.flatten(), t_tensor.flatten()], dim=1)

    # Evaluate the model
    u_tensor = model(combined_input).reshape(x_tensor.shape)

    cax = ax.pcolormesh(x_grid, y_grid, u_tensor.detach().numpy(), cmap='plasma', vmin=0, vmax=1)
    fig.colorbar(cax)

    def update(t):
        # Create a combined tensor for x, y, and the current t value
        t_tensor = torch.full_like(x_tensor, fill_value=t, dtype=torch.float32)
        combined_input = torch.stack([x_tensor.flatten(), y_tensor.flatten(), t_tensor.flatten()], dim=1)

        # Evaluate the model
        u_tensor = model(combined_input).reshape(x_tensor.shape)

        # Plot the field u(x, y) for this t value
        cax.set_array(u_tensor.detach().numpy().flatten())
        
        ax.set_title(f"Field u(x, y) at t={t:.2f}")


    # Create animation
    ani = FuncAnimation(fig, update, frames=t_values, blit=False)

    # Display the animation or save it as a file
    
    # plt.show()

    # Uncomment the line below to save the animation as a file
    ani.save(os.path.join(basedir,'u_field.gif'), writer='imagemagick')