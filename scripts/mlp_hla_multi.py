import os
import sys
sys.path.append("..")  # add top folder to path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, precision_score
import torch
import impepdom
from impepdom import models


model = models.MultilayerPerceptron(num_hidden_layers=2, hidden_layer_size=100)
dataset = impepdom.PeptideDataset(
    hla_allele='HLA-A01:01',
    padding='flurry',
    toy=False)
    
results = impepdom.hyperparam_grid_search(model, dataset)
for res in results:
    print(res)