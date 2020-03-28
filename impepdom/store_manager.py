import os
import pickle
from datetime import datetime

import torch


STORE_PATH = os.path.join(os.getcwd(), '../store')

def get_save_path(model, hla_allele, train_fold_idx):
    '''
    Saves model trained state and history of training and validation metrics.
    Model outputs are uniquely stored in the a string composed of:
        Model name,
        Hidden layer size,
        HLA allele,
        Datetime of experiment,
        Folds used to train on.
    Example: "/store/mlp_2x100_a01:01/200319_174308/train_012/".

    Parameters
    ----------
    hla_allele: string
        Name of the MHC allele used for training
    
    train_fold_idx: list
        Indices of folds used for training

    Returns
    ----------
    folder: string
        Path inside STORE_PATH to folder storing model information
    '''

    name = model.get_my_name()
    allele = hla_allele[4:].lower()  # crop the "hla-" part since they all share the same 
    train_folds = list_to_str(train_fold_idx)

    # storage format is model_name-hla_allele/datetime_of_model_save/train-fold_indices
    dt = datetime.now().strftime('%y%m%d_%H%M%S')  # format is yymmdd_hhmmss
    folder = '{0}_{1}/{2}/train_{3}'.format(name, allele, dt, train_folds)

    path_to_folder = os.path.join(STORE_PATH, folder)
    os.makedirs(path_to_folder, exist_ok=True)

    return folder

def load_trained_model(model, folder):
    '''
    Load from information from /store 
    
    Parameters
    ----------
    model: nn.Module
        Initialized class of the model to be loaded
    folder: string
        Path inside STORE_PATH to folder storing trained model information

    Returns
    ----------
    model: string
        Return trained torch model
    
    train_history: dict
        Dictionary of training history
    '''

    # fetch trained torch model
    model_path = os.path.join(STORE_PATH, folder, 'torch_model')
    model.load_state_dict(torch.load(model_path))
    model.eval()  # set to inference mode

    # fetch training history
    train_history_path = os.path.join(STORE_PATH, folder, 'train_history')
    infile = open(train_history_path,'rb')
    train_history = pickle.load(infile)
    infile.close()

    return model, train_history

def pickle_dump(data, folder, filename):
    path = os.path.join(STORE_PATH, folder, filename)
    out = open(path, 'wb')
    pickle.dump(data, out)

def list_to_str(ls):
    _str = ''.join(sorted([str(c) for c in ls]))
    return _str

def extract_date(folder):
    '''
    Get the date out of folder path.
    "mlp_a01:01/200319_174308/train_012/" -> "200319_174308"
    '''

    path = folder
    path = folder[folder.find('/') + 1:]
    path = folder[:path.find('/')]
    
    return path
