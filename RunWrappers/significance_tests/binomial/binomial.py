"""
This script is intended to implement the binomial tests that specifically tests 
if the two binomial distributions are different for a particular metric
"""
from utils import compute_metrics
import numpy as np
import math
import scipy.stats

def compute_z_score(p1: float, p2: float, n1: int, n2: int):
    p_hat = (n1*p1 + n2*p2) / (n1 + n2)
    return (p1-p2) / math.sqrt(p_hat*(1-p_hat)*(1/n1 + 1/n2))

def compute_binomial_p_value(
    classifier_A_predictions: np.ndarray,
    classifier_B_predictions: np.ndarray, 
    y_test: np.ndarray,
    is_multiclass: bool
):
    assert classifier_A_predictions.size == classifier_B_predictions.size
    assert classifier_A_predictions.size == y_test.size
    p_values = {
        # metric name -> p-value
    }
    # compute initial metrics
    A_metrics = compute_metrics(
        y_test, classifier_A_predictions, 
        prefix="test", is_multiclass=is_multiclass
    )
    B_metrics = compute_metrics(
        y_test, classifier_B_predictions, 
        prefix="test", is_multiclass=is_multiclass
    )
    # null: p1 = p2 (where p1 is the greater metric)
    # alternative: p1 \= p2
    metric_names = A_metrics.keys()
    for m in metric_names:
        z = compute_z_score(
            A_metrics[m], B_metrics[m], 
            classifier_A_predictions.size, classifier_B_predictions.size)
        p_values[m] = scipy.stats.norm.sf(abs(z))*2

    return p_values
