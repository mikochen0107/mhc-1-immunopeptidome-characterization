def weighted_harmonic_mean(var_1, var_2, beta=1):  # should make it generalizable to many variables
    '''
    Harmonic mean for two hyperparameters with weighting.
    
    Parameters
    ----------
    var_1, var_2: int or ndarray
        Variables to consider
        
    beta: float, optional
        Importance of `var_2` relative to `var_1`.
        If beta == 1, this function is equivalent to `scipy.stats.hmean()`
    '''
    
    return (1 + beta**2) * np.multiply(var_1, var_2) / (beta**2 * var_1 + var_2)

def get_best_hyperparams(file, padding='end'):
    '''
    Extract the best hyperparameters for a model with harmonic weighting

    Parameters
    ----------
    file: string
        Name of the CSV file contraining hyperparam results
    '''

    path = '../store/hyperparams'

    file_end = file.find('.csv')
    file_begin = max(file.find('mlp'), file.find('cnn'))
    all_name = file[file_end-6:file_end-2] + (':' if file.find(':') == -1 else '') + file[file_end-2:file_end]
    allele = 'HLA-' + all_name.upper() # change to appropriate name

    df = pd.read_csv(path + '/' + file)
    correct_padding = (df['padding'] == padding)
    df = df[correct_padding].reset_index()
    idx = (df['min_auc'].notna() & df['mean_ppv'].notna())
    
    metric_1, metric_2 = np.array(df['min_auc'][idx]), np.array(df['mean_ppv'][idx])
    beta = 1  # how much the second metric should be weighted compared to the first
    w_hmean = weighted_harmonic_mean(metric_1, metric_2, beta=0.6)
    
    best_3_rows = (-w_hmean).argsort()[:3] # for top 3 rows with best harmonic mean value
    # best_3_rows = (-sts.hmean([df['min_auc'][idx], df['mean_ppv'][idx]])).argsort()[:3]
    
    batch_sizes = list(df['batch_size'][best_3_rows].astype('int'))
    batch_counter = Counter(batch_sizes)
    batch_sz = batch_counter.most_common(1)[0][0]
    
    hyperparams = {
        'hla_allele': allele, 
        'padding': padding,
        'batch_size': batch_sz, 
        'num_epochs': int(np.mean(df['num_epochs'][best_3_rows])),
        'learning_rate': float(np.mean(df['learning_rate'][best_3_rows])),

        'min_auc': list(metric_1[best_3_rows]),
        'mean_ppv': list(metric_2[best_3_rows]),
    }

    if 'mean_pcc' in df.columns:
        hyperparams['mean_pcc'] = list(df['mean_pcc'][best_3_rows])
    if 'dropout_input' in df.columns and 'dropout_hidden' in df.columns:
        hyperparams['dropout_input'] = float(np.mean(df['dropout_input'][best_3_rows]))
        hyperparams['dropout_hidden'] = float(np.mean(df['dropout_hidden'][best_3_rows]))
    else:
        hyperparams['dropout_input'] = 0.65  # stolen from MLP end
        hyperparams['dropout_hidden'] = 0.46
        
    hyperparams['conv'] = False
    if 'conv' in df.columns:
        if df['conv'][0] == 'True':
            hyperparams['conv'] = True
            
    if 'num_conv_layers' in df.columns and 'conv_filt_sz' in df.columns	and 'conv_stride' in df.columns:
        hyperparams['num_conv_layers'] = int(np.max(df['num_conv_layers'][best_3_rows]))
        hyperparams['conv_filt_sz'] = int(np.max(df['conv_filt_sz'][best_3_rows]))
        hyperparams['conv_stride'] = int(np.max(df['conv_stride'][best_3_rows]))
    else:
        hyperparams['num_conv_layers'] = 1  # default params for conv
        hyperparams['conv_filt_sz'] = 5
        hyperparams['conv_stride'] = 2

    return hyperparams

def make_trained_model(hyperparams):
    results = []

    impepdom.time_tracker.reset_timer() 

    print('working with allele', hyperparams['hla_allele'])
    dataset = impepdom.PeptideDataset(
        hla_allele=hyperparams['hla_allele'],
        padding=hyperparams['padding'],
        toy=False)

    save_folder, baseline_metrics, _ = impepdom.run_experiment(
        model_type='MultilayerPerceptron',  # passing model type here
        dataset=dataset,
        train_fold_idx=[0, 1, 2, 3, 4],
        learning_rate=hyperparams['learning_rate'],
        num_epochs=hyperparams['num_epochs'],
        batch_size=hyperparams['batch_size'],

        show_output=True,
        dropout_input=hyperparams['dropout_input'],
        dropout_hidden=hyperparams['dropout_hidden'],

        conv=hyperparams['conv'],
        num_conv_layers=hyperparams['num_conv_layers'],
        conv_filt_sz=hyperparams['conv_filt_sz'],
        conv_stride=hyperparams['conv_stride']
    )

    model = impepdom.models['MultilayerPerceptron'](
        dropout_input=hyperparams['dropout_input'],
        dropout_hidden=hyperparams['dropout_hidden'],

        conv=hyperparams['conv'],
        num_conv_layers=hyperparams['num_conv_layers'],
        conv_filt_sz=hyperparams['conv_filt_sz'],
        conv_stride=hyperparams['conv_stride']
    )
    
    # to load a trained model, you would need to initialize the model outside
    # smth like (untrained) `model = impepdom.models.MultilayerPerceptron(num_hidden_layers=2, hidden_layer_size=100)`
    # but specify params exactly the same as in hyperparam_search
                                                                    # get from csv file (column "model")
    trained_model, _ = impepdom.load_trained_model(model, save_folder)

    return trained_model, save_folder