from typing import Any, Optional, Sequence, Type, Union, Literal

from torchmetrics import Metric
# from torchmetrics.classification import ConfusionMatrix
# from avalanche.evaluation.metrics import Accuracy, ConfusionMatrix   
# from avalanche.evaluation.metric_results import MetricValue
# from avalanche.evaluation.metric_utils import get_metric_name
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