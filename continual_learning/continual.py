from avalanche.benchmarks import SplitMNIST, PermutedMNIST
from avalanche.models import IncrementalClassifier
from avalanche.models import MultiHeadClassifier
from avalanche.models import SimpleCNN
from avalanche.models import as_multitask
from avalanche.benchmarks.datasets import FashionMNIST, MNIST
from avalanche.benchmarks.generators import nc_benchmark, ni_benchmark
import torch
import torchvision
import os
from utils import utils

from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive
from avalanche.training.plugins import EarlyStoppingPlugin

from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics, class_accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, \
confusion_matrix_metrics, disk_usage_metrics, StreamClassAccuracy, StreamAccuracy
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin

# --- CONFIG
device = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)


def run_continual(train, test, num_class, model_name, lr=0.001,train_epochs=10, experiences=5):

    benchmark = ni_benchmark(
        train_dataset=train, 
        test_dataset=test,
        n_experiences=experiences, 
        task_labels=False,
        balance_experiences=True,
    )

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        class_accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True),
        forgetting_metrics(experience=True, stream=True),
        cpu_usage_metrics(experience=True),
        confusion_matrix_metrics(num_classes=num_class, save_image=False, stream=True),
        disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        StreamClassAccuracy(classes=list(range(0,num_class))),
        StreamAccuracy(),
        loggers=[InteractiveLogger()],
        strict_checks=False
    )
    
    model = utils.make_model_pretrained(model_name=model_name, num_class=num_class)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()
    cl_strategy = Naive(
        model, optimizer, criterion,
        train_mb_size=100, train_epochs=train_epochs, eval_mb_size=100,
        plugins=[EarlyStoppingPlugin(patience=5, val_stream_name='train')],
        evaluator=eval_plugin
    )
    
    results = []
    print('Starting experiment...')
    print("Training...")
    for experience in benchmark.train_stream:
        print("Start of experience ", experience.current_experience)
        #experiences have an ID that denotes its position in the stream
        # this is used only for logging (don't rely on it for training!)
        eid = experience.current_experience
        # for classification benchmarks, experiences have a list of classes in this experience
        clss = experience.classes_in_this_experience
        task = experience.task_label
        print(f"EID={eid}, classes={clss}, task={task}")
        # the experience provides a dataset
        print(f"data: {len(experience.dataset)} samples")
        
        cl_strategy.train(experience)
        print('Training completed')
        
        print('Computing accuracy on the whole test set')
        results.append(cl_strategy.eval(benchmark.test_stream))
    

    return results