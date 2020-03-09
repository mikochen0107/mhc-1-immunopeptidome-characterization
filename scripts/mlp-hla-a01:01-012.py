import os
import sys
sys.path.append("..")  # add top folder to path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import roc_auc_score
import impepdom
import torch


hla_allele = 'HLA-A01:01'
fold_idx = [0, 1, 2]
hla_a01_01 = impepdom.PeptideDataset(hla_allele, padding='after2', toy=False)
peploader = hla_a01_01.get_peptide_dataloader(fold_idx)
mlp = impepdom.MultilayerPerceptron()

folder = impepdom.train_nn(mlp, peploader, hla_allele, fold_idx, save_results=True)

data, targets = hla_a01_01.get_fold([3])
predictions = mlp(torch.tensor(data).float())
pred_path = os.path.join(folder, 'pred-3')
outfile = open(pred_path, 'wb')
pickle.dump(predictions, outfile)
outfile.close()

preds = np.round(predictions.detach().numpy())  # get binary prediction
print("AUC ROC score:", roc_auc_score(torch.tensor(targets).float().detach().numpy(), preds))
