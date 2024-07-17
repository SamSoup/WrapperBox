from transformers import TrainerCallback


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience: int, metric: str, mode: str = "max"):
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.best_metric = None
        self.patience_counter = 0

    def on_evaluate(self, args, state, control, **kwargs):
        metric_value = kwargs["metrics"].get(self.metric)
        if metric_value is None:
            return

        if (
            self.best_metric is None
            or (self.mode == "max" and metric_value > self.best_metric)
            or (self.mode == "min" and metric_value < self.best_metric)
        ):
            self.best_metric = metric_value
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            control.should_training_stop = True
