
import torch.nn as nn
# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
def get_loss(output,rating):
    criterion = nn.MSELoss()
    return criterion