import os
import time
from datetime import datetime

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train_nn(model, peploader, criterion, optimizer, scheduler=None, num_epochs=25, learning_rate=1e-2, save_results=True):
    '''
    Train neural network using gradient descent and backprop

    Parameters
    ----------
    model: nn.Module
        Defined neural network
    
    Returns
    ----------
    train_history: dict
        Dictionary of defined metrics for train and val sets for each epoch
    '''

    since = time.time()
    train_history = init_train_hist()
    
    for e in range(num_epochs):
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
        
        print('epoch {0} \t train loss {1} \t time {2} s'.format(e, round(running_loss / len(peploader), 5), round(time.time() - since, 5)))
    
    return train_history

def init_train_hist():
    metrics = ['loss', 'acc', 'auc', 'auc0.1', 'pcc']
    train_history = {
        'train': {},
        'val': {}
    }

    for metric in metrics:
        train_history['train'][metric] = []
        train_history['val'][metric] = []

    return train_history