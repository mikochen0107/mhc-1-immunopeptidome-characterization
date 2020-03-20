from sklearn.metrics import precision_score, roc_auc_score
import numpy as np

def eval(y_true, y_pred):



	AUC = roc_auc_score(y_true, y_pred)

	AUC_01 = roc_auc_score(y_true, y_pred, max_fpr=0.1)

	top_100_idx = np.argsort(y_pred)[-100:]
	y_true_100 = [y_true[i] for i in top_100_idx]
	y_pred_100 = [y_pred[i] for i in top_100_idx]
	PPV_100 = precision_score(y_true_100, y_pred_100)

    return AUC, AUC_01, PPV_100