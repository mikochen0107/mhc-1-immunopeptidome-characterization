import os
import time
from datetime import datetime

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


STORE_PATH = '../store'

def train_nn(model, peploader, hla_allele, fold_idx, criterion=None, optimizer=None, scheduler=None, num_epochs=25, learning_rate=1e-2, save_results=True):
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

    if save_results:
        folder = save_model(model, hla_allele, fold_idx)
        return folder
    return None

def save_model(model, hla_allele, fold_idx):
    '''
    Saves model trained state and history of training and validation metrics.

    Parameters
    ----------
    model: nn.Module
        Trained neural network

    hla_allele: string
        Name of the MHC allele used for training
    
    fold_idx: list
        Indices of folds used for training
    '''

    name = model.get_my_name()
    allele = hla_allele[4:].lower()  # crop the "hla-" part since they all share the same 
    train_folds = ''.join(sorted([str(c) for c in fold_idx]))

    # storage format is model_name - hla_allele - fold indices - date, time of model save
    dt = datetime.now().strftime('%y%m%d%H%M%S')
    folder = '{0}-{1}-{2}-'.format(name, allele, train_folds) + dt
    os.makedirs(os.path.join(STORE_PATH, folder), exist_ok=True)
    filename = os.path.join(STORE_PATH, folder, 'model_state_dict')
    torch.save(model.state_dict(), filename)

    return os.path.join(STORE_PATH, folder)

