from utils import compute_metrics
import numpy as np

def sample_with_replacement(X: np.ndarray, size=1000):
    indices = np.random.choice(np.arange(X.shape[0]), size, replace=True)
    return X[indices], indices

def compute_bootstrap_p_value(
    classifier_A_predictions: np.ndarray,
    classifier_B_predictions: np.ndarray, 
    y_test: np.ndarray,
    is_multiclass: bool,
    size=1000, iterations=10000, seed=42
):
    np.random.seed(seed) # for reproducability
    assert classifier_A_predictions.size == classifier_B_predictions.size
    assert classifier_A_predictions.size == y_test.size
    deltas = {
        # metric name -> {delta:<>, count:<>, p-value:<>}
    }
    # compute initial deltas
    A_metrics = compute_metrics(
        y_test, classifier_A_predictions, 
        prefix="test", is_multiclass=is_multiclass
    )
    B_metrics = compute_metrics(
        y_test, classifier_B_predictions, 
        prefix="test", is_multiclass=is_multiclass
    )
    metric_names = A_metrics.keys()
    for m in metric_names:
        # gain should always be positive, so take absolute value
        # original delta, number of times where δ(x(i)) > 2δ(x)
        deltas[m] = {
            'delta': abs(A_metrics[m] - B_metrics[m]),
            'count': 0,
            'is_A_greater': A_metrics[m] >= B_metrics[m] # A > B, else B > A
        }
    for _ in range(iterations):
        y_boot, indices = sample_with_replacement(y_test, size=size)
        A_metrics = compute_metrics(
            y_boot, classifier_A_predictions[indices], 
            prefix="test", is_multiclass=is_multiclass
        )
        B_metrics = compute_metrics(
            y_boot, classifier_B_predictions[indices], 
            prefix="test", is_multiclass=is_multiclass
        )
        for m in metric_names:
            if deltas[m]['is_A_greater']:
                boot_delta = A_metrics[m] - B_metrics[m]
            else:
                boot_delta = B_metrics[m] - A_metrics[m]
            if boot_delta > 2*deltas[m]['delta']:
                deltas[m]['count'] += 1
    # estimate the p-values
    for m in metric_names:
        deltas[m]['p-value'] = 1 - deltas[m]['count'] / iterations

    return deltas
