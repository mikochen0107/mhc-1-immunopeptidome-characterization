import os
import sys
sys.path.append("..")  # add top folder to path

import impepdom

# os.listdir('../datasets/MHC_I_el_allele_specific')
hla_alleles = [
    'HLA-B44:03',  # >>> Khoi 1
    'HLA-B08:01',
    'HLA-A01:01',
    'HLA-B15:01',  # >>> Khoi 1
    'HLA-A02:01',  # >>> Khoi 2
    'HLA-A03:01',  
    'HLA-B07:02',  # <<< Khoi 2
    'HLA-A24:02',  # >>> Michael
    'HLA-B27:05', 
    'HLA-A68:01'   # <<< Michael
]

hla_alleles_khoi_1 = hla_alleles[:4]
hla_alleles_khoi_2 = hla_alleles[4:7]
hla_alleles_michael = hla_alleles[7:]

model = impepdom.models.MultilayerPerceptron(num_hidden_layers=2, hidden_layer_size=100)

for i, hla_allele in enumerate(hla_alleles):
    print('working with allele {0} out of {1}'.format(i + 1, len(hla_alleles)))
    dataset = impepdom.PeptideDataset(
        hla_allele=hla_alleles_michael,  # change allele here
        padding='flurry',
        toy=False)
        
    results = impepdom.hyperparam_grid_search(
        model,
        dataset,
        epochs=[5, 10, 15],
        batch_sizes=[32, 64, 128],
        learning_rates=[5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
    )

    for res in results:
        print(res)