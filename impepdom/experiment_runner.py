import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


STORE_PATH = '../store'

def run_experiment(
    model, hla_allele='HLA-A01:01', train_fold_idx=[0, 1, 2], val_fold_idx=[3], padding='after2',
    criterion=None, optimizer=None, scheduler=None, num_epochs=25, learning_rate=1e-2, show_output=True
):
    '''
    Run a neural network training on specified train and validation set, with parameters.

    Parameters
    ----------
    model: nn.Module
        Defined neural network

    hla_allele: string, optional
        Name of the folder corresponding to MHC I allele

    train_fold_idx: list, optional
        List of number (from 0 to 4) to specify training folds

    val_fold_idx: list, optional
        List of number (from 0 to 4) to specify validation folds

    criterion: nn.Loss, optional
    optimizer: torch.optim, optional
    scheduler
    num_epochs: int, optional
    learning_rate: float, optional

    show_output: bool
        Show output of training process (everything will be saved anyway)

    Returns
    ----------
    save_path: string
        Relative path where the cache is located
    '''

    # obtain torch dataloader for batch learning
    dataset = impepdom.PeptideDataset(hla_allele, padding=padding, toy=False)
    peploader = {}
    peploader['train'] = dataset.get_peptide_dataloader(train_fold_idx)
    peploader['val'] = dataset.get_peptide_dataloader(val_fold_idx)

    # set up optimization criterion and optimization algorithm
    if criterion == None:
        criterion = nn.BCELoss()
    if optimizer == None:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train the model, collect data
    model, train_history = impepdom.train_nn(
        model=model,
        peploader=peploader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        save_results=True)

    # save model
    save_folder = get_save_path(hla_allele, train_fold_idx)
    torch.save(model.state_dict(), os.path.join(save_folder, 'torch_model'))
    
    # save training history
    pickle_dump(train_history, save_folder, 'train_history')

    # save validation predictions
    val = {}
    data, val['target'] = hla_a01_01.get_fold(val_fold_idx)
    val['pred'] = mlp(torch.tensor(data).float())
    pickle_dump(val, save_folder, 'validation_' + list_to_str(val_fold_idx))

def get_save_path(hla_allele, train_fold_idx):
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

def list_to_str(ls):
    str = ''.join(sorted([str(c) for c in ls]))
    return str

def pickle_dump(data, save_folder, filename):
    path = os.path.join(save_folder, filename)
    out = open(path, 'wb')
    pickle.dump(data, out)