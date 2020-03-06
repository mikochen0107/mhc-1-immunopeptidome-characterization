import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train_nn(model, peploader, criterion=None, optimizer=None, scheduler=None, num_epochs=25, learning_rate=1e-2):
    '''
    Train neural network using gradient descent and backprop

    Parameters
    ----------
    model: nn.Module
        Defined neural network
    '''

    if criterion == None:
        criterion = nn.MSELoss()
    if optimizer == None:
        optimizer = optim.Adam([
            {'params': model.parameters()}
        ], lr=learning_rate)
    
    for _ in range(num_epochs):
        running_loss = 0
        for peps, targets in peploader:
            # training pass
            optimizer.zero_grad()
            
            output = model(peps)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print('Training loss: {0}'.format(running_loss / len(trainloader)))
