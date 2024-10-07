from typing import Any, Optional, Literal

import torchmetrics
import torch
from avalanche.evaluation import PluginMetric, Metric, GenericPluginMetric
from avalanche.evaluation.metrics.mean import Mean
from avalanche.evaluation import Metric

class BalancedAccuracy(torchmetrics.Metric):
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

class AverageBadDecisionMetric(Metric[float]):
    """
    Average Bad Decision (ABD) metric.
    
    This metric tracks false positives (FP) and false negatives (FN) across
    experiences and batches to compute an average metric indicating how often 
    incorrect predictions are made relative to the change between batches.
    
    Each time `result` is called, this metric emits the ABD based on all 
    predictions made since the last reset.
    """
    
    def __init__(self):
        """Creates an instance of the AverageBadDecision metric.

        This metric tracks the FP and FN differences across batches
        to compute the ABD. The running metric can be retrieved using
        the `result` method, and the state can be reset using `reset`.
        """
        self._mean_FP = Mean()
        self._mean_FN = Mean()
        self._previous_FP = 0
        self._previous_FN = 0
        self._current_batch = 0

    @torch.no_grad()
    def update(self, predicted_y: torch.Tensor, true_y: torch.Tensor) -> None:
        """
        Update the running ABD metric with the current batch predictions.

        :param predicted_y: The predicted labels (either logits or plain labels).
        :param true_y: The ground truth labels (either logits or plain labels).
        """
        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        if len(true_y) != len(predicted_y):
            raise ValueError("Size mismatch between true_y and predicted_y tensors.")

        # If predictions are logits, convert to labels
        if len(predicted_y.shape) > 1:
            predicted_y = torch.max(predicted_y, 1)[1]

        if len(true_y.shape) > 1:
            true_y = torch.max(true_y, 1)[1]

        # Compute False Positives (FP) and False Negatives (FN)
        FP = float(torch.sum((predicted_y == 1) & (true_y == 0)))
        FN = float(torch.sum((predicted_y == 0) & (true_y == 1)))

        # Update Mean with the differences between current and previous values
        if self._current_batch > 0:
            self._mean_FP.update(abs(FP - self._previous_FP))
            self._mean_FN.update(abs(FN - self._previous_FN))

        # Update previous batch values and increment batch count
        self._previous_FP = FP
        self._previous_FN = FN
        self._current_batch += 1

    def result(self) -> float:
        """Return the average ABD (FP + FN difference) across batches."""
        FP_mean = self._mean_FP.result()
        FN_mean = self._mean_FN.result()
        return FP_mean + FN_mean

    def reset(self) -> None:
        """Reset the state of the ABD metric."""
        self._mean_FP = Mean()
        self._mean_FN = Mean()
        self._previous_FP = 0
        self._previous_FN = 0
        self._current_batch = 0

    def __str__(self):
        return "AverageBadDecision"

class AverageBadDecisionPluginMetric(GenericPluginMetric[float, AverageBadDecisionMetric]):
    """
    Base class for all AverageBadDecision plugin metrics.
    """
    
    def __init__(self, reset_at, emit_at, mode):
        """
        Creates the AverageBadDecision plugin.

        :param reset_at: When to reset the metric ('stream', 'experience', 'epoch').
        :param emit_at: When to emit the metric ('stream', 'experience', 'epoch').
        :param mode: The mode of the metric ('train', 'eval').
        """
        super().__init__(AverageBadDecisionMetric(), reset_at=reset_at, emit_at=emit_at, mode=mode)

    def reset(self, strategy=None) -> None:
        """Resets the underlying metric."""
        self._metric.reset()

    def result(self, strategy=None) -> float:
        """Returns the result of the underlying metric."""
        return self._metric.result()

    def update(self, strategy) -> None:
        """Updates the underlying metric based on model output and targets."""
        self._metric.update(strategy.mb_output, strategy.mb_y)

class AverageBadDecision(AverageBadDecisionPluginMetric):
    """
    Computes the Average Bad Decision metric at the end of the stream during evaluation.
    """

    def __init__(self):
        """
        Creates an instance of the AverageBadDecision plugin metric.
        """
        super(AverageBadDecision, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval"
        )

    def __str__(self):
        return "AverageBadDecision"

class OCMMetric(Metric[float]):
    """
    Overall Changing Mistakes (OCM) metric.

    This metric tracks False Positive Rate (FPR) and False Negative Rate (FNR) 
    across tasks and computes the Average FPR (AFPR), Average FNR (AFNR),
    and OCM based on the formula:

        OCM = AFNR + AFPR

    where AFNR and AFPR measure how FPR and FNR change from task to task.
    """

    def __init__(self):
        """Creates an instance of the OCM metric.

        This metric tracks FPR and FNR differences across tasks
        to compute AFPR, AFNR, and the final OCM.
        """
        self.FPR_per_task = []
        self.FNR_per_task = []
        self._current_task = 0

    @torch.no_grad()
    def update(self, predicted_y: torch.Tensor, true_y: torch.Tensor) -> None:
        """
        Update the OCM metric with the current batch predictions.

        :param predicted_y: The predicted labels (either logits or plain labels).
        :param true_y: The ground truth labels (either logits or plain labels).
        """
        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        if len(true_y) != len(predicted_y):
            raise ValueError("Size mismatch between true_y and predicted_y tensors.")

        # Convert logits to labels if necessary
        if len(predicted_y.shape) > 1:
            predicted_y = torch.max(predicted_y, 1)[1]

        if len(true_y.shape) > 1:
            true_y = torch.max(true_y, 1)[1]

        # Compute TP, TN, FP, FN
        TP = float(torch.sum((predicted_y == 1) & (true_y == 1)))
        TN = float(torch.sum((predicted_y == 0) & (true_y == 0)))
        FP = float(torch.sum((predicted_y == 1) & (true_y == 0)))
        FN = float(torch.sum((predicted_y == 0) & (true_y == 1)))

        # Calculate FPR and FNR
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        FNR = FN / (TP + FN) if (TP + FN) > 0 else 0

        # Store FPR and FNR for the current task
        self.FPR_per_task.append(FPR)
        self.FNR_per_task.append(FNR)
        self._current_task += 1

    def result(self) -> float:
        """Compute the Overall Changing Mistakes (OCM) metric."""
        T = len(self.FPR_per_task)
        if T < 2:
            return float('nan')  # OCM is undefined for less than 2 tasks

        # Calculate AFPR and AFNR
        AFPR = sum(self.FPR_per_task[i-1] - self.FPR_per_task[i] for i in range(1, T)) / (T - 1)
        AFNR = sum(self.FNR_per_task[i-1] - self.FNR_per_task[i] for i in range(1, T)) / (T - 1)

        # Calculate OCM as the sum of AFPR and AFNR
        OCM = AFPR + AFNR
        return OCM

    def reset(self) -> None:
        """Reset the OCM metric."""
        self.FPR_per_task.clear()
        self.FNR_per_task.clear()
        self._current_task = 0

    def __str__(self):
        return "OCM"

class OCMPluginMetric(GenericPluginMetric[float, OCMMetric]):
    """
    Plugin class for Overall Changing Mistakes (OCM) metric.

    This plugin allows tracking the OCM during training or evaluation.
    """

    def __init__(self, reset_at, emit_at, mode):
        """
        Creates an instance of the OCM plugin metric.

        :param reset_at: When to reset the metric ('stream', 'experience', 'epoch').
        :param emit_at: When to emit the metric ('stream', 'experience', 'epoch').
        :param mode: The mode of the metric ('train', 'eval').
        """
        super().__init__(OCMMetric(), reset_at=reset_at, emit_at=emit_at, mode=mode)

    def reset(self, strategy=None) -> None:
        """Resets the underlying OCM metric."""
        self._metric.reset()

    def result(self, strategy=None) -> float:
        """Returns the result of the underlying OCM metric."""
        return self._metric.result()

    def update(self, strategy) -> None:
        """Updates the underlying OCM metric based on model output and targets."""
        self._metric.update(strategy.mb_output, strategy.mb_y)

class OCM(OCMPluginMetric):
    """
    Computes the OCM metric at the end of the stream during evaluation.
    """

    def __init__(self):
        """
        Creates an instance of the OCM plugin metric.
        """
        super(OCM, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval"
        )

    def __str__(self):
        return "OCM"
    
class AverageForgettingMetric(Metric[float]):
    """
    Average Forgetting (AF) metric.
    
    This metric tracks the accuracy across experiences to compute an average
    forgetting metric indicating how much the model forgets over time.
    
    Each time `result` is called, this metric emits the AF based on all 
    predictions made since the last reset.
    """
    
    def __init__(self):
        """Creates an instance of the AverageForgetting metric.

        This metric tracks the accuracy differences across experiences
        to compute the AF. The running metric can be retrieved using
        the `result` method, and the state can be reset using `reset`.
        """
        self._accuracies = {}
        self._max_accuracies = {}
        self._current_experience = 0

    @torch.no_grad()
    def update(self, predicted_y: torch.Tensor, true_y: torch.Tensor) -> None:
        """
        Update the running AF metric with the current batch predictions.

        :param predicted_y: The predicted labels (either logits or plain labels).
        :param true_y: The ground truth labels (either logits or plain labels).
        """
        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        if len(true_y) != len(predicted_y):
            raise ValueError("Size mismatch between true_y and predicted_y tensors.")

        # If predictions are logits, convert to labels
        if len(predicted_y.shape) > 1:
            predicted_y = torch.max(predicted_y, 1)[1]

        if len(true_y.shape) > 1:
            true_y = torch.max(true_y, 1)[1]

        # Compute accuracy
        correct = float(torch.sum(predicted_y == true_y))
        total = float(len(true_y))
        accuracy = correct / total

        # Update accuracies for the current experience
        if self._current_experience not in self._accuracies:
            self._accuracies[self._current_experience] = []
        self._accuracies[self._current_experience].append(accuracy)

        # Update max accuracies
        if self._current_experience not in self._max_accuracies:
            self._max_accuracies[self._current_experience] = accuracy
        else:
            self._max_accuracies[self._current_experience] = max(self._max_accuracies[self._current_experience], accuracy)

    def result(self) -> float:
        """Return the average forgetting (max accuracy - final accuracy) across experiences."""
        forgetting = 0.0
        num_experiences = len(self._accuracies)

        for exp in range(1, num_experiences + 1):
            max_acc = self._max_accuracies[exp]
            final_acc = self._accuracies[exp][-1]
            forgetting += max_acc - final_acc

        if num_experiences > 0:
            forgetting /= num_experiences

        return forgetting

    def reset(self) -> None:
        """Reset the state of the AF metric."""
        self._accuracies = {}
        self._max_accuracies = {}
        self._current_experience = 0

    def __str__(self):
        return "AverageForgetting"

from avalanche.evaluation import GenericPluginMetric

class AverageForgettingPluginMetric(GenericPluginMetric[float, AverageForgettingMetric]):
    """
    Base class for all AverageForgetting plugin metrics.
    """
    
    def __init__(self, reset_at, emit_at, mode):
        """
        Creates the AverageForgetting plugin.

        :param reset_at: When to reset the metric ('stream', 'experience', 'epoch').
        :param emit_at: When to emit the metric ('stream', 'experience', 'epoch').
        :param mode: The mode of the metric ('train', 'eval').
        """
        super().__init__(AverageForgettingMetric(), reset_at=reset_at, emit_at=emit_at, mode=mode)

    def reset(self, strategy=None) -> None:
        """Resets the underlying metric."""
        self._metric.reset()

    def result(self, strategy=None) -> float:
        """Returns the result of the underlying metric."""
        return self._metric.result()

    def update(self, strategy) -> None:
        """Updates the underlying metric based on model output and targets."""
        self._metric.update(strategy.mb_output, strategy.mb_y)
    
class AverageForgetting(AverageForgettingPluginMetric):
    """
    Computes the Average Forgetting metric at the end of the stream during evaluation.
    """

    def __init__(self):
        """
        Creates an instance of the AverageForgetting plugin metric.
        """
        super(AverageForgetting, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval"
        )

    def __str__(self):
        return "AverageForgetting"