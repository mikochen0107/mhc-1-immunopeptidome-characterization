import os
import time
from datetime import datetime

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


PATH = '../store'

def train_nn(model, peploader, criterion=None, optimizer=None, scheduler=None, num_epochs=25, learning_rate=1e-2):
    '''
    Train neural network using gradient descent and backprop

    Parameters
    ----------
    model: nn.Module
        Defined neural network
    '''

    since = time.time()

    if criterion == None:
        criterion = nn.BCELoss()
    if optimizer == None:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for _ in range(num_epochs):
        running_loss = 0

        for item in peploader:
            pep = item['peptide']
            target = item['target'].view(-1, 1)
            optimizer.zero_grad()

            output = model(pep.float())
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print('Training loss: {0}, time elapsed: {1} s'.format(running_loss / len(peploader), time.time() - since))

    save_model(model)

def save_model(model):
    name = model.get_my_name()
    dt = datetime.now().strftime("{}-%y%m%d-%H%M%S".format(name))
    os.mkdir(os.path.join(PATH, dt))
    filename = os.path.join(PATH, dt, 'model_state_dict')
    torch.save(model.state_dict(), filename)

