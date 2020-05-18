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

hla_alleles_test = ['HLA-A01:01']
impepdom.time_tracker.reset_timer()  # start counting time

for i, hla_allele in enumerate(hla_alleles_test):  # change allele list here
    print(impepdom.time_tracker.now() + 'working with allele {0} out of {1}'.format(i + 1, len(hla_alleles_test)))  # change allele list here
    
    dataset = impepdom.PeptideDataset(
        hla_allele=hla_allele,  
        padding='flurry',
        toy=False)
        
    best_config = impepdom.hyperparam_search(
        model_type='MultilayerPerceptron',
        dataset=dataset,
        max_epochs=15,
        batch_sizes=[32],
        learning_rates=[7e-4],

        dropout_input_list=[0.20, 0.15],
        dropout_hidden_list=[0.45, 0.35],
        conv_flags=[True],
        num_conv_layers_list=[2],
        conv_filt_sz_list=[5],
        conv_stride_list=[1],
    )

    print(best_config)