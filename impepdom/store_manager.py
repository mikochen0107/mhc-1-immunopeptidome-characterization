import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import torch

import impepdom.metrics


STORE_PATH = os.path.join(os.getcwd(), '../store')

def get_save_path(model, hla_allele, train_fold_idx, which_model):
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

    which_model: string
        Datetime of model run to write into. Format: 'mlp_2x10_a01:01/200327_212136'

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
    if which_model == None:
        folder = '{0}_{1}/{2}/train_{3}'.format(name, allele, dt, train_folds)
    else:
        folder = os.path.join(which_model, 'train_{0}'.format(train_folds))

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
    model_path = os.path.join(STORE_PATH, folder, 'best_model')
    model.load_state_dict(torch.load(model_path))
    model.eval()  # set to inference mode

    # fetch training history
    train_history_path = os.path.join(STORE_PATH, folder, 'train_history')
    infile = open(train_history_path,'rb')
    train_history = pickle.load(infile)
    infile.close()

    return model, train_history

def update_hyperparams_store(results_store):
    '''
    Update .csv table of experiments with different hyperparameters.

    Parameters
    ----------
    results_store: dict
        Dictionary containing model name, hyperparam settings, and evaluation metrics
    '''

    # {
    #     'model': folder,
    #     'mean_' + _eval: np.mean(cross_eval),
    #     'min_' + _eval: np.min(cross_eval),
    #     'max_' + _eval: np.max(cross_eval),
    #     'batch_size': batch_size,
    #     'num_epochs': num_epochs,
    #     'learning_rate': learning_rate,
     #           }
    
    new_hyperparam_vals = get_hyperparams_store_template()
    for res in results_store:
        for key in new_hyperparam_vals:
            if key in res:
                new_hyperparam_vals[key].append(res[key])
            else:
                new_hyperparam_vals[key].append(None)

    folder = results_store[0]['model']
    new_hyperparam_df = pd.DataFrame(new_hyperparam_vals)
    hyperparams_path = os.path.join(STORE_PATH, 'hyperparams', folder[:folder.find('/')] + '.csv')

    if not os.path.exists(hyperparams_path):
        new_hyperparam_df.to_csv(hyperparams_path)
    else:
        hyperparam_df = pd.read_csv(hyperparams_path, index_col=0)
        hyperparam_df = pd.concat([hyperparam_df, new_hyperparam_df], ignore_index=True)
        hyperparam_df.sort_values(by=['mean_auc', 'min_auc'], inplace=True, ascending=[False, False], ignore_index=True)
        hyperparam_df.to_csv(hyperparams_path)

def update_metrics(model_run_time):
    pass

def fetch_best_hyperparams(model_name, _eval):
    '''
    Retrieve best hyperparameter settings for a model based on evaluation metric.

    Parameters
    ----------
    model_name: string
        Model name from models.Model.get_my_name(). Format: 'mlp_2x100'

    _eval: string
        Evaluation metric. Options: 'acc', 'auc', 'auc_01', 'ppv'

    Returns
    ----------
    hyperparams: dict
        Dictionary containing hyperameter settings
    '''

    pass

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

def extract_which_model(folder):
    '''
    Get folder upper of a chosen model training to write
    multiple results of cross-validation.
    '''

    path = folder
    path = path[:path.rfind('train')-1] 

    return path

def get_hyperparams_store_template():
    '''
    Return an empty pre-configured dictionary template to store hyperparameters. 

    Returns
    ----------
    hyperparams: dict
        Dictionary of model hyperparameters configuration and results.
    '''

    metrics = impepdom.metrics.METRICS
    desc_stats = impepdom.metrics.DESC_STATS

    hyperparams = {
        'model': [],
        'padding': [],
        'batch_size': [],
        'num_epochs': [],
        'learning_rate': [],
        'optimizer': [],
        'scheduler': [],
    }

    # initilize space for metrics
    for metric in metrics:
        for desc_stat in desc_stats:
            hyperparams[desc_stat[0] + '_' + metric] = []

    return hyperparams
