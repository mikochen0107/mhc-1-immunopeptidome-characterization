from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from scipy.stats import pearsonr
import numpy as np


METRICS = ['acc', 'f1', 'auc', 'auc_01', 'ppv', 'ppv_100']
DESC_STATS = [('mean', np.mean), ('min', np.min), ('max', np.max)]

def auc(y_true, y_proba):
	return roc_auc_score(y_true, y_proba)

def auc_01(y_true, y_proba):
	return roc_auc_score(y_true, y_proba, max_fpr=0.1)

def ppv(y_true, y_proba):
	'''
	PPV is calculated by sorting the y_true based on y_proba (the prediction scores), and calculating
	the true positive rate in the first n elements in the sorted y_true array, where n refers to the 
	number of positives in y_true. 
	'''

	num_of_1s = np.sum(y_true == 1)
	sorted_y_true = np.flip([x for _, x in sorted(zip(y_proba, y_true))])
	ppv_score = np.sum(sorted_y_true[:num_of_1s] == 1) / num_of_1s

	return ppv_score


def pcc(y_true, y_proba):
	'''
	PCC is the Pearson correlation coefficient. The correlation is calculated between the true labels (y_true) and the predicted probabilities (y_proba).
	The function from scipy is used, where the first returned value is the coefficient and the second value is the p-value. We only want the first one.
	'''

	pcc_score = pearsonr(y_true, y_proba)[0] 

	return pcc_score


def ppv_100(y_true, y_proba):
	'''
	Function that calculates the ppv score for the top 100 predictions (instead of number of positives). 
	If the number of positives is smaller than 100, then the function defaults to the normal ppv function.
	'''
	
	num_of_1s = np.sum(y_true == 1)
	if num_of_1s < 100:
		return ppv(y_true, y_proba)
	sorted_y_true = np.flip([x for _, x in sorted(zip(y_proba, y_true))])
	ppv_100_score = np.sum(sorted_y_true[:100] == 1) / 100  # percentage of positives in top-rated peptides

	return ppv_100_score

def acc(y_true, y_pred):
	return accuracy_score(y_true, y_pred)

def f1(y_true, y_pred):
	return f1_score(y_true, y_pred) 