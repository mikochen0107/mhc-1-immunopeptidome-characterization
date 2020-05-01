import os
import sys
sys.path.append("..")  # add top folder to path

import impepdom

hla_alleles = os.listdir('../datasets/MHC_I_el_allele_specific')

model = impepdom.models.MultilayerPerceptron(num_hidden_layers=2, hidden_layer_size=100)

for i, hla_allele in enumerate(hla_alleles):
    print('working with allele {0} out of {1}'.format(i + 1, len(hla_alleles)))
    dataset = impepdom.PeptideDataset(
        hla_allele=hla_allele,
        padding='end',
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