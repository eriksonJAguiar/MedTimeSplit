from typing import Any, Optional, Sequence, Type, Union, Literal

from torchmetrics import Metric
from avalanche.evaluation import PluginMetric
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

class AverageBadDecisionMetric(PluginMetric[float]):
    def __init__(self):
        super().__init__()
        self.FP_per_experience = []
        self.FN_per_experience = []
        self.FP_per_stream = []
        self.FN_per_stream = []
        self.current_experience_FP = 0
        self.current_experience_FN = 0

    def reset(self) -> None:
        """Reset metric state for a new stream."""
        self.FP_per_experience.clear()
        self.FN_per_experience.clear()
        self.FP_per_stream.clear()
        self.FN_per_stream.clear()
        self.current_experience_FP = 0
        self.current_experience_FN = 0
        print("Metric reset")

    def update(self, preds, targets) -> None:
        """Update FP and FN for each mini-batch."""
        FP = ((preds == 1) & (targets == 0)).sum().item()
        FN = ((preds == 0) & (targets == 1)).sum().item()
        print(f"Updating metrics: FP={FP}, FN={FN}")
        self.current_experience_FP += FP
        self.current_experience_FN += FN

    def before_experience(self, strategy: "SupervisedTemplate") -> None:
        """Reset current experience metrics before each experience."""
        self.current_experience_FP = 0
        self.current_experience_FN = 0
        print("Before experience reset")

    def after_experience(self, strategy: "SupervisedTemplate") -> None:
        """At the end of an experience, store the accumulated FP and FN."""
        self.FP_per_experience.append(self.current_experience_FP)
        self.FN_per_experience.append(self.current_experience_FN)
        # Also accumulate results into stream-level metrics
        self.FP_per_stream.append(self.current_experience_FP)
        self.FN_per_stream.append(self.current_experience_FN)
        print(f"After experience: FP={self.current_experience_FP}, FN={self.current_experience_FN}")

    def result(self) -> float:
        """Calculate the ABD based on all experiences in a stream (stream=True) or just for one experience (experience=True)."""
        if len(self.FP_per_experience) < 2:
            print("Not enough data to compute the stream metric")
            return float('nan')  # Not enough data to compute the metric

        # For experience-level: only calculate ABD for current experience
        T = len(self.FP_per_experience)
        FP_T = sum(abs(self.FP_per_experience[i] - self.FP_per_experience[i - 1]) for i in range(1, T)) / (T - 1)
        FN_T = sum(abs(self.FN_per_experience[i] - self.FN_per_experience[i - 1]) for i in range(1, T)) / (T - 1)
        ABD = FP_T + FN_T
        print(f"Result: ABD={ABD}")
        return ABD

    def stream_result(self) -> float:
        """Calculate the ABD based on the entire stream."""
        if len(self.FP_per_stream) < 2:
            print("Not enough data to compute the stream metric")
            return float('nan')  # Not enough data to compute the metric

        T = len(self.FP_per_stream)
        FP_T = sum(abs(self.FP_per_stream[i] - self.FP_per_stream[i - 1]) for i in range(1, T)) / (T - 1)
        FN_T = sum(abs(self.FN_per_stream[i] - self.FN_per_stream[i - 1]) for i in range(1, T)) / (T - 1)
        ABD = FP_T + FN_T
        return ABD

    def __str__(self):
        return "AverageBadDecision"

class OCMMetric(PluginMetric[float]):
    def __init__(self):
        super().__init__()
        self.FPR_per_experience = []
        self.FNR_per_experience = []
        self.FPR_per_stream = []
        self.FNR_per_stream = []
        self.current_experience_FPR = 0
        self.current_experience_FNR = 0

    def reset(self) -> None:
        """Reset metric state for a new stream."""
        self.FPR_per_experience.clear()
        self.FNR_per_experience.clear()
        self.FPR_per_stream.clear()
        self.FNR_per_stream.clear()
        self.current_experience_FPR = 0
        self.current_experience_FNR = 0

    def update(self, preds, targets) -> None:
        """Update FPR and FNR for each mini-batch."""
        TP = ((preds == 1) & (targets == 1)).sum().item()
        TN = ((preds == 0) & (targets == 0)).sum().item()
        FP = ((preds == 1) & (targets == 0)).sum().item()
        FN = ((preds == 0) & (targets == 1)).sum().item()

        # Calculate FPR and FNR
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        FNR = FN / (TP + FN) if (TP + FN) > 0 else 0

        self.current_experience_FPR += FPR
        self.current_experience_FNR += FNR

    def before_experience(self, strategy: "SupervisedTemplate") -> None:
        """Reset current experience metrics before each experience."""
        self.current_experience_FPR = 0
        self.current_experience_FNR = 0

    def after_experience(self, strategy: "SupervisedTemplate") -> None:
        """At the end of an experience, store the accumulated FPR and FNR."""
        self.FPR_per_experience.append(self.current_experience_FPR)
        self.FNR_per_experience.append(self.current_experience_FNR)
        # Also accumulate results into stream-level metrics
        self.FPR_per_stream.append(self.current_experience_FPR)
        self.FNR_per_stream.append(self.current_experience_FNR)

    def result(self) -> float:
        """Calculate the OCM based on all experiences in a stream (stream=True) or just for one experience (experience=True)."""
        if len(self.FPR_per_experience) < 2:
            return float('nan')  # Not enough data to compute the metric

        T = len(self.FPR_per_experience)
        AFPR = sum((self.FPR_per_experience[i - 1] - self.FPR_per_experience[i]) for i in range(1, T)) / (T - 1)
        AFNR = sum((self.FNR_per_experience[i - 1] - self.FNR_per_experience[i]) for i in range(1, T)) / (T - 1)
        OCM = AFPR + AFNR
        return OCM

    def stream_result(self) -> float:
        """Calculate the OCM based on the entire stream."""
        if len(self.FPR_per_stream) < 2:
            return float('nan')  # Not enough data to compute the metric

        T = len(self.FPR_per_stream)
        AFPR = sum((self.FPR_per_stream[i - 1] - self.FPR_per_stream[i]) for i in range(1, T)) / (T - 1)
        AFNR = sum((self.FNR_per_stream[i - 1] - self.FNR_per_stream[i]) for i in range(1, T)) / (T - 1)
        OCM = AFPR + AFNR
        return OCM

    def __str__(self):
        return "OCM"