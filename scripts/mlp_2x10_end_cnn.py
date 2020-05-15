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
        learning_rates=[1e-3],

        dropout_input_list=[0.25],
        dropout_hidden_list=[0.55],
        conv_flags=[True],
        num_conv_layers_list=[2],
        conv_filt_sz_list=[5],
        conv_stride_list=[1],
    )

    print(best_config)

# impepdom.time_tracker.reset_timer()  # start counting time

# for i, hla_allele in enumerate(hla_alleles_khoi_1):  # change allele list here
#     print(impepdom.time_tracker.now() + 'working with allele {0} out of {1}'.format(i + 1, len(hla_alleles_khoi_1)))  # change allele list here
    
#     dataset = impepdom.PeptideDataset(
#         hla_allele=hla_allele,  
#         padding='end',
#         toy=False
#     )
        
#     best_config = impepdom.hyperparam_search(
#         model_type='MultilayerPerceptron',
#         dataset,
#         max_epochs=15,
#         batch_sizes=[32, 64],
#         learning_rates=[5e-4, 1e-3, 5e-3, 1e-2],
#         dropout_input_list=[0.6, 0.85],
#         dropout_hidden_list=[0.4, 0.65],
#         conv_flags=[True],
#         num_conv_layers_list=[1, 2],
#         conv_filt_sz_list=[3, 5],
#         conv_stride_list=[1, 2],
#     )

#     print(best_config)