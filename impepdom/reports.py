import torch

import impepdom.metrics

def get_predictions(model, X_test_store):
    y_pred_store = {}
    for key, X_test in X_test_store.items():
        y_pred_store[key] = predict(model, X_test)
        
    return y_pred_store

def predict(model, X_test):
    y_proba = model(torch.tensor(X_test, dtype=torch.float)).detach().numpy().reshape(-1)
    
    return y_proba

'''
Fetch performance metrics for epitope and MS ligands test sets
'''
def get_epi_metrics(y_test_store, y_pred_store):
    scores_store = {}
    
    for key in y_pred_store.keys():
        scores_store[key] = {
            'auc': impepdom.metrics.auc(y_test_store[key], y_pred_store[key]),
            'f_rank': impepdom.metrics.f_rank(y_test_store[key], y_pred_store[key])
        }
        
    return scores_store

def get_ms_metrics(y_test_store, y_pred_store):
    scores_store = {}
    
    for key in y_pred_store.keys():
        scores_store['HLA-B08:01'] = {
            'auc': impepdom.metrics.auc(y_test_store[key], y_pred_store[key]),
            'auc_01': impepdom.metrics.auc_01(y_test_store[key], y_pred_store[key]),
            'ppv': impepdom.metrics.ppv(y_test_store[key], y_pred_store[key])
        }
        
    return scores_store

'''
Generate and save performance table for epitope and MS ligands test sets
'''
def generate_epi_report(scores_store, filename):
    pep_col = []
    auc_col = []
    frank_col = []
    
    for key, scores in scores_store.items():
        pep_col.append(key)
        auc_col.append(scores['auc'])
        frank_col.append(scores['f_rank'])
        
    report = pd.DataFrame({'Epitope': pep_col, 'AUC': auc_col, 'FRANK': frank_col})
    
    report_root = '../store/reports/test_set/epi'
    report.to_csv(os.path.join(report_root, filename))
    
    return report

def generate_ms_metrics(scores_store, filename):
    mhc_col = []
    auc_col = []
    auc_01_col = []
    ppv_col = []
    
    for key, scores in scores_store.items():
        mhc_col.append(key)
        auc_col.append(scores['auc'])
        auc_01_col.append(scores['auc_01'])
        ppv_col.append(scores['ppv'])
        
    report = pd.DataFrame({'MHC': mhc_col, 'AUC': auc_col, 'AUC0.1': auc_01_col, 'PPV': ppv_col})
    
    report_root = '../store/reports/test_set/ms'
    report.to_csv(os.path.join(report_root, filename))
    
    return report