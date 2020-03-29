# Peptide ligand classification with multilayer perceptrons
## Description
To be added...

## Installation
#### Step 1: System pre-requisites
If you're using macOS, install `pipenv` with Homebrew
```
$ brew install pipenv
```
More documentation: https://github.com/pypa/pipenv

#### Step 2: Clone repository and install requirements
```
$ git clone https://github.com/mikochen0107/mhc-1-immunopeptidome-characterization.git
$ cd mhc-1-immunopeptidome-characterization
$ pipenv install
```

## Usage
### Scenario 1: Train a model
#### Step 1: Create a script
Create a script in `/scripts` or a Jupyter notebook in `/notebooks`

#### Step 2: Example code
```
import sys
sys.path.append("..")  # add top folder to path
import impepdom

model = impepdom.models.MultilayerPerceptron(num_hidden_layers=2, hidden_layer_size=100)
dataset = impepdom.PeptideDataset(
    hla_allele='HLA-A01:01',
    padding='flurry',
    toy=True)

folder, baseline_metrics, _ = impepdom.run_experiment(
    model,
    dataset,
    train_fold_idx=[1, 2, 3],
    val_fold_idx=[0],
    learning_rate=2e-3,
    num_epochs=5,
    batch_size=128)

trained_model, train_history = impepdom.load_trained_model(model, folder)
impepdom.plot_train_history(train_history, baseline_metrics)
```

#### Step 3: Run code (if it is a script)
```
$ pipenv shell
$ cd scripts
$ python <NAME_OF_SCRIPT>
```

### Scenario 2: Hyperparameter tuning
To be added...

## Formats
To be added...
