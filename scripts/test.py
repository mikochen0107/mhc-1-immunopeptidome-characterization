import os
import sys
sys.path.append("..")  # add top folder to path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import impepdom
import pickle


hla_allele = 'HLA-A01:01'
fold_idx = [0, 1, 2]
hla_a01_01 = impepdom.TrainPeptideDataset(hla_allele, padding='after2', toy=True)
peploader = hla_a01_01.get_peptide_dataloader(fold_idx)
mlp = impepdom.MultilayerPerceptron()

folder = impepdom.train_nn(mlp, peploader, hla_allele, fold_idx, save_results=True)

predictions = mlp(torch.tensor(hla_a01_01.get_fold([3])[0]).float())
print(folder)
pred_path = os.path.join(folder, 'pred-3')
outfile = open(pred_path, 'wb')
pickle.dump(predictions, outfile)
outfile.close()

infile = open(pred_path,'rb')
print(pickle.load(infile))
infile.close()

