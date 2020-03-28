import sys
sys.path.append("..")  # add top folder to path

import impepdom


model = impepdom.models.MultilayerPerceptron(num_hidden_layers=2, hidden_layer_size=100)
dataset = impepdom.PeptideDataset(
    hla_allele='HLA-A01:01',
    padding='flurry',
    toy=False)
    
results = impepdom.hyperparam_grid_search(
    model,
    dataset,
    epochs=[1],
    batch_sizes=[32],
    learning_rates=[5e-3],
)

for res in results:
    print(res)