
from avalanche.models import IncrementalClassifier
from avalanche.models import MultiHeadClassifier
from avalanche.models import SimpleCNN
from avalanche.models import as_multitask
from avalanche.benchmarks import nc_benchmark, ni_benchmark
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.logging import TextLogger
import torch
import torchvision
import os
from utils import utils
from utils.metrics import OCMMetric, AverageBadDecisionMetric

from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive
from avalanche.training.supervised.strategy_wrappers_online import OnlineNaive
from avalanche.training.plugins import EarlyStoppingPlugin

from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets
from avalanche.benchmarks.scenarios.online import split_online_stream

from avalanche.evaluation.metrics import forgetting_metrics, forward_transfer_metrics, \
accuracy_metrics, class_accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, \
confusion_matrix_metrics, disk_usage_metrics, StreamClassAccuracy, StreamAccuracy, \
StreamBWT, StreamForwardTransfer, ExperienceBWT, ExperienceForwardTransfer, \
bwt_metrics, forward_transfer_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.benchmarks.utils import concat_datasets
from avalanche.training.templates import SupervisedTemplate


device = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Device: {device}")

class Cumulative(SupervisedTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = None  # cumulative dataset

    def train_dataset_adaptation(self, **kwargs):
        super().train_dataset_adaptation(**kwargs)
        curr_data = self.experience.dataset
        if self.dataset is None:
            self.dataset = curr_data
        else:
            self.dataset = concat_datasets([self.dataset, curr_data])
        self.adapted_dataset = self.dataset.train()


def run_continual(train, test, num_class, model_name, lr=0.001, train_epochs=10, experiences=4):
    """running continual learning

    Args:
        train (Dataset): train dataset splited.
        test (Dataset): train dataset splited.
        num_class (int): number of class in database.
        model_name (str): deep architecture name.
        lr (float, optional): learning rate. Defaults to 0.001.
        train_epochs (int, optional): Number of epoches. Defaults to 10.
        experiences (int, optional): number of experiences. Defaults to 4.

    Returns:
        _type_: _description_
    """
    benchmark = ni_benchmark(
        train_dataset=train, 
        test_dataset=test,
        n_experiences=experiences, 
        task_labels=False,
        shuffle=True, 
        balance_experiences=True
    )
    
    # train_stream_online = split_online_stream(
    #     original_stream=benchmark.train_stream,
    #     experience_size=len(train)//16,
    #     drop_last=True
    # )
    
    # test_stream_online = split_online_stream(
    #     original_stream=benchmark.test_stream,
    #     experience_size=len(train)//80,
    #     drop_last=True
    # )

    eval_plugin = EvaluationPlugin(
        #accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        #class_accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        #loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        #timing_metrics(epoch=True),
        #forgetting_metrics(experience=True, stream=True),
        #bwt_metrics(experience=True, stream=True),
        #forward_transfer_metrics(experience=True, stream=True),
        #cpu_usage_metrics(experience=True),
        AverageBadDecisionMetric(),
        OCMMetric(),
        #confusion_matrix_metrics(num_classes=num_class, save_image=False, stream=True),
        #disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        #StreamClassAccuracy(classes=list(range(0,num_class))),
        # StreamAccuracy(),
        # StreamBWT(),
        # StreamForwardTransfer(),
        # ExperienceForwardTransfer(),
        # ExperienceBWT(),
        #bwt_metrics(experience=True, stream=True),
        #forward_transfer_metrics(experience=True, stream=True),
        loggers=[InteractiveLogger(), TextLogger()],
        strict_checks=False
    )
    
    model = utils.make_model_pretrained(model_name=model_name, num_class=num_class)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.001)
    criterion = CrossEntropyLoss()
    cl_strategy = Naive(
        model, optimizer, criterion,
        train_mb_size=train_epochs, train_epochs=train_epochs,
        evaluator=eval_plugin,
        device=device
    )
    # cl_strategy = OnlineNaive(
    #     model, optimizer, criterion,
    #     train_mb_size=16, eval_mb_size=16,
    #     train_passes=train_epochs,
    #     eval_every=train_epochs,
    #     evaluator=eval_plugin,
    #     #plugins=[eval_plugin],
    #     device=device
    # )
    # cl_strategy = Cumulative(
    #     model, optimizer, criterion,
    #     train_mb_size=128, train_epochs=train_epochs, eval_mb_size=128,
    #     plugins=[EarlyStoppingPlugin(patience=5, val_stream_name='train')],
    #     evaluator=eval_plugin,
    #     device=device
    # )
        
    results = []
    print('Starting experiment...')
    print("Training...")
    #for experience, exp_test in zip(train_stream_online, test_stream_online):
    for experience in benchmark.train_stream:
        print("Start of experience ", experience.current_experience)
        # experiences have an ID that denotes its position in the stream
        # this is used only for logging (don't rely on it for training!)
        eid = experience.current_experience
        # for classification benchmarks, experiences have a list of classes in this experience
        clss = experience.classes_in_this_experience
        print(experience.dataset)
        #clss = [target for _, target in experience.dataset]
        #task = experience.task_label
        #print(f"EID={eid} classes={len(clss)} task={task}")
        print(f"EID={eid}")
        # the experience provides a dataset
        print(f"data: {len(experience.dataset)} samples")
        
        try:
            cl_strategy.train(experience)
        except KeyError as e:
            print(f"KeyError on experience {experience.current_experience}: {e}")
            print(f"Data in previous: {cl_strategy.evaluator.previous}")
            print(f"Data in initial: {cl_strategy.evaluator.initial}")
        
        print('Training completed')
        
        print('Computing accuracy on the whole test set')
        metrics_performance = cl_strategy.eval(benchmark.test_stream)
        
        forward_transfer_score = metrics_performance.get('ForwardTransfer/Stream', 0)
        print(f"Forward Transfer Score: {forward_transfer_score}")
        
        keys = list(metrics_performance.keys())
        line = {}
        for i in range(len(keys)):
            keys_slice = keys[i].split("/")
            metric = keys_slice[0]
            if metric.find("ClassAcc") != -1:
                metric = metric + "_" + keys_slice[-1]
            
            phase = keys_slice[1].split("_")[0]
            metric = metric + "_" + phase
            line["Experiment"] = str(eid)
            line[metric] = metrics_performance[keys[i]]
            
        
        results.append(line)
        #results.append(metrics_performance)
    
#     forgetting_scores = [result['StreamForgetting_Stream'] for result in results if 'StreamForgetting_Stream' in result]

#     # Calculate the average forgetting score across all experiences
#   average_forgetting = sum(forgetting_scores) / len(forgetting_scores)
  
    print(results)
    #for res in enumerate(results)
    
    return results