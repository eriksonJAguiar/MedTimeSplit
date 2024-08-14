from typing import Any, Optional, Sequence, Type, Union, Literal

from torchmetrics import Metric
from torchmetrics.classification import ConfusionMatrix
from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metrics import Accuracy, ConfusionMatrix   
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation.metric_utils import get_metric_name
import torch

class BalancedAccuracy(Metric):
    def __init__(
        self,
        task: Literal["binary", "multiclass"],
        num_classes: Optional[int] = None, 
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.task = task
        self.num_classes = num_classes

        self.add_state("tp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        if self.task == "multiclass":
            preds_one_hot = torch.nn.functional.one_hot(preds, num_classes=self.num_classes)
            target_one_hot = torch.nn.functional.one_hot(target, num_classes=self.num_classes)
            for i in range(self.num_classes):
                self.tp += torch.sum((preds_one_hot[:, i] == 1) & (target_one_hot[:, i] == 1)).float()
                self.fp += torch.sum((preds_one_hot[:, i] == 1) & (target_one_hot[:, i] == 0)).float()
                self.fn += torch.sum((preds_one_hot[:, i] == 0) & (target_one_hot[:, i] == 1)).float()
                self.tn += torch.sum((preds_one_hot[:, i] == 0) & (target_one_hot[:, i] == 0)).float()

        else:  # Binary case
            self.tp += torch.sum((preds == 1) & (target == 1)).float()
            self.fp += torch.sum((preds == 1) & (target == 0)).float()
            self.fn += torch.sum((preds == 0) & (target == 1)).float()
            self.tn += torch.sum((preds == 0) & (target == 0)).float()

    def compute(self) -> torch.Tensor:
        tpr = self.tp / (self.tp + self.fn)  # True Positive Rate
        tnr = self.tn / (self.tn + self.fp)  # True Negative Rate
        
        balanced_accuracy = (tpr + tnr) / 2
        return balanced_accuracy

class Prevalence(Metric):
    def __init__(self,
                task: Literal["binary", "multiclass"],
                num_classes: Optional[int] = None, 
                **kwargs: Any
            ) -> Metric:
        super().__init__(**kwargs)
        self.task = task
        self.num_classes = num_classes
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")
        
        if self.task == "multiclass":
            preds_one_hot = torch.nn.functional.one_hot(preds, num_classes=self.num_classes)
            target_one_hot = torch.nn.functional.one_hot(target, num_classes=self.num_classes)
            tp_aux, tn_aux, fp_aux, fn_aux = 0, 0, 0, 0
            for i in range(self.num_classes):
                tp_aux += torch.sum((preds_one_hot[:, i] == 1) & (target_one_hot[:, i] == 1)).item()
                fn_aux += torch.sum((preds_one_hot[:, i] == 0) & (target_one_hot[:, i] == 1)).item()
            
            self.tp += tp_aux // self.num_classes
            self.fn += fn_aux // self.num_classes
        
        else:
            self.tp += torch.sum((preds_one_hot[:, i] == 1) & (target_one_hot[:, i] == 1)).item()
            self.fn += torch.sum((preds_one_hot[:, i] == 0) & (target_one_hot[:, i] == 1)).item()
        
        self.total += target.numel()

    def compute(self) -> torch.Tensor:
        
        return (self.tp.float() + self.fn.float()) / self.total

class Bias(Metric):
    def __init__(self,
                task: Literal["binary", "multiclass"],
                num_classes: Optional[int] = None, 
                **kwargs: Any
            ) -> Metric:
        super().__init__(**kwargs)
        self.task = task
        self.num_classes = num_classes
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")
        
        if self.task == "multiclass":
            preds_one_hot = torch.nn.functional.one_hot(preds, num_classes=self.num_classes)
            target_one_hot = torch.nn.functional.one_hot(target, num_classes=self.num_classes)
            tp_aux, tn_aux, fp_aux, fn_aux = 0, 0, 0, 0
            for i in range(self.num_classes):
                tp_aux += torch.sum((preds_one_hot[:, i] == 1) & (target_one_hot[:, i] == 1)).item()
                fp_aux += torch.sum((preds_one_hot[:, i] == 1) & (target_one_hot[:, i] == 0)).item()
                fn_aux += torch.sum((preds_one_hot[:, i] == 0) & (target_one_hot[:, i] == 1)).item()
                tn_aux += torch.sum((preds_one_hot[:, i] == 0) & (target_one_hot[:, i] == 0)).item()
            
            self.tp += tp_aux // self.num_classes
            self.tn += tn_aux // self.num_classes
            self.fp += fp_aux // self.num_classes
            self.fn += fn_aux // self.num_classes
        
        else:
            self.tp += torch.sum((preds_one_hot[:, i] == 1) & (target_one_hot[:, i] == 1)).item()
            self.fp += torch.sum((preds_one_hot[:, i] == 1) & (target_one_hot[:, i] == 0)).item()
            self.fn += torch.sum((preds_one_hot[:, i] == 0) & (target_one_hot[:, i] == 1)).item()
            self.tn += torch.sum((preds_one_hot[:, i] == 0) & (target_one_hot[:, i] == 0)).item()
        
        self.total += target.numel()

    def compute(self) -> torch.Tensor:
        
        tpr = torch.divide(self.tp.float(), (self.tp + self.fn)).float()
        tnr = torch.divide(self.tn.float(), (self.tn + self.fp)).float()
        prevanlence = (self.tp.float() + self.fn.float()) / self.total
        
        return tpr * prevanlence + (1 - tnr) * (1 - prevanlence)

class A(PluginMetric[float]):
    """
    This metric will return a `float` value after
    each training epoch
    """

    def __init__(self):
        """
        Initialize the metric
        """
        super().__init__()

        self._accuracy_metric = Accuracy()

    def reset(self, **kwargs) -> None:
        """
        Reset the metric
        """
        self._accuracy_metric.reset()

    def result(self, **kwargs) -> float:
        """
        Emit the result
        """
        return self._accuracy_metric.result()

    def after_training_iteration(self, strategy: 'PluggableStrategy') -> None:
        """
        Update the accuracy metric with the current
        predictions and targets
        """
        # task labels defined for each experience
        task_labels = strategy.experience.task_labels
        if len(task_labels) > 1:
            # task labels defined for each pattern
            task_labels = strategy.mb_task_id
        else:
            task_labels = task_labels[0]
            
        self._accuracy_metric.update(strategy.mb_output, strategy.mb_y, 
                                     task_labels)

    def before_training_epoch(self, strategy: 'PluggableStrategy') -> None:
        """
        Reset the accuracy before the epoch begins
        """
        self.reset()

    def after_training_epoch(self, strategy: 'PluggableStrategy'):
        """
        Emit the result
        """
        return self._package_result(strategy)
            
    def _package_result(self, strategy):
        """Taken from `GenericPluginMetric`, check that class out!"""
        metric_value = self.accuracy_metric.result()
        add_exp = False
        plot_x_position = strategy.clock.train_iterations

        if isinstance(metric_value, dict):
            metrics = []
            for k, v in metric_value.items():
                metric_name = get_metric_name(
                    self, strategy, add_experience=add_exp, add_task=k)
                metrics.append(MetricValue(self, metric_name, v,
                                           plot_x_position))
            return metrics
        else:
            metric_name = get_metric_name(self, strategy,
                                          add_experience=add_exp,
                                          add_task=True)
            return [MetricValue(self, metric_name, metric_value,
                                plot_x_position)]

    def __str__(self):
        """
        Here you can specify the name of your metric
        """
        return "Top1_Acc_Epoch"