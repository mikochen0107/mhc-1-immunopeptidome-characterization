import os
import sys
sys.path.append("..")  # add top folder to path

import impepdom

# os.listdir('../datasets/MHC_I_el_allele_specific')
hla_alleles = [
    'HLA-B44:03',  
    'HLA-B08:01',  # >>> Khoi 1
    'HLA-A01:01',
    'HLA-B15:01',  
    'HLA-A02:01', 
    'HLA-A03:01',  # <<< Khoi 1
    'HLA-B07:02',  # >>> Khoi 2 <<<
    'HLA-A24:02',  
    'HLA-B27:05',  # >>> Michael
    'HLA-A68:01'   # <<< Michael
]

hla_alleles_khoi_1 = hla_alleles[1:6]
hla_alleles_khoi_2 = hla_alleles[6:7]
hla_alleles_michael = hla_alleles[8:]
hla_alleles_test = ['HLA-A01:01']
hla_alleles_rerun = ['HLA-B44:03', 'HLA-A24:02']

impepdom.time_tracker.reset_timer()  # start counting time

for i, hla_allele in enumerate(hla_alleles_rerun):  # change allele list here
    print(impepdom.time_tracker.now() + 'working with allele {0} out of {1}'.format(i + 1, len(hla_alleles_rerun)))  # change allele list here
    
    model = impepdom.models.MultilayerPerceptron(num_hidden_layers=2, hidden_layer_size=100)  # reset model
    dataset = impepdom.PeptideDataset(
        hla_allele=hla_allele,  
        padding='flurry',
        toy=False
    )
        
    best_config = impepdom.hyperparam_search(
        model,
        dataset,
        max_epochs=15,
        batch_sizes=[32, 64, 128],
        learning_rates=[5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
    )

    print(best_config)