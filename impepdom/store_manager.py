import os
import pickle
from datetime import datetime

import torch


STORE_PATH = '../store'

def get_save_path(model, hla_allele, train_fold_idx):
    '''
    Saves model trained state and history of training and validation metrics.

    Parameters
    ----------
    hla_allele: string
        Name of the MHC allele used for training
    
    train_fold_idx: list
        Indices of folds used for training

    Returns
    ----------
    path_to_folder: string
        Relative path to folder storing model information
    '''

    name = model.get_my_name()
    allele = hla_allele[4:].lower()  # crop the "hla-" part since they all share the same 
    train_folds = list_to_str(train_fold_idx)

    # storage format is model_name - hla_allele - fold indices - date, time of model save
    dt = datetime.now().strftime('%y%m%d%H%M%S')
    folder = '{0}_{1}_{2}_'.format(name, allele, train_folds) + dt
    path_to_folder = os.path.join(STORE_PATH, folder)
    os.makedirs(path_to_folder, exist_ok=True)

    return path_to_folder

def load_trained_model(model, save_folder):
    '''
    Load from information from /store 
    
    Parameters
    ----------
    model: nn.Module
        Initialized class of the model to be loaded
    save_folder: string
        Path to folder storing trained model

    Returns
    ----------
    model: string
        Return trained torch model
    
    train_history: dict
        Dictionary of training history
    '''

    # fetch trained torch model
    model_path = os.path.join(save_folder, 'torch_model')
    model.load_state_dict(torch.load(model_path))
    model.eval()  # set to inference mode

    # fetch training history
    train_history_path = os.path.join(save_folder, 'train_history')
    infile = open(train_history_path,'rb')
    train_history = pickle.load(infile)
    infile.close()

    return model, train_history

def pickle_dump(data, save_folder, filename):
    path = os.path.join(save_folder, filename)
    out = open(path, 'wb')
    pickle.dump(data, out)

def list_to_str(ls):
    _str = ''.join(sorted([str(c) for c in ls]))
    return _str

