
from avalanche.models import IncrementalClassifier
from avalanche.models import MultiHeadClassifier
from avalanche.models import SimpleCNN
from avalanche.models import as_multitask
from avalanche.benchmarks import nc_benchmark, ni_benchmark
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.logging import TextLogger
import torch
from utils import utils
from utils.metrics import OCM, AverageBadDecision


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
from avalanche.benchmarks import SplitImageNet

from utils.partitioning import load_database_federated_continous, load_database_federated_continousSplit, load_database_federated_continousPermuted
from utils import utils
import pandas as pd
import numpy as np
import os
import json
from torch.optim import Adam
from torch.nn import CrossEntropyLoss


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


# def __train_continous(train, num_class, lr, train_epochs, experience_size):

def run_continual(train, test, model, optimizer, criterion, train_epochs=10, train_mb_size=16, experiences=4, is_train=None):
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
    
    train_stream_online = split_online_stream(
        original_stream=benchmark.train_stream,
        experience_size=len(train)//16,
        drop_last=True
    )
    
    test_stream_online = split_online_stream(
        original_stream=benchmark.test_stream,
        experience_size=len(test)//16,
        drop_last=True
    )

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        #class_accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True),
        forgetting_metrics(experience=True, stream=True),
        bwt_metrics(experience=True, stream=True),
        #AverageForgetting(),
        #forward_transfer_metrics(experience=True, stream=True),
        #cpu_usage_metrics(experience=True),
        AverageBadDecision(),
        OCM(),
        #confusion_matrix_metrics(num_classes=num_class, save_image=False, stream=True),
        #disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        #StreamClassAccuracy(classes=list(range(0,num_class))),
        #StreamAccuracy(),
        # StreamBWT(),
        # StreamForwardTransfer(),
        # ExperienceForwardTransfer(),
        # ExperienceBWT(),
        #bwt_metrics(experience=True, stream=True),
        #forward_transfer_metrics(experience=True, stream=True),
        loggers=[InteractiveLogger(), TextLogger()],
        strict_checks=False
    )
    
    cl_strategy = OnlineNaive(
        model, optimizer, criterion,
        train_mb_size=train_mb_size, eval_mb_size=train_mb_size,
        train_passes=train_epochs,
        eval_every=1,
        evaluator=eval_plugin,
        device=device
    )
        
    results = []
    print('Starting experiment...')
    print("Training...")
    for experience, exp_test in zip(train_stream_online, test_stream_online):
    #for experience in benchmark.train_stream:
        print("Start of experience ", experience.current_experience)
        print(f"Experience {experience.current_experience} - Number of samples: {len(experience.dataset)}")
        # experiences have an ID that denotes its position in the stream
        eid = experience.current_experience
        print(f"EID={eid}")
        # the experience provides a dataset
        try:
            cl_strategy.train(experience)
        except KeyError as e:
            print(f"KeyError on experience {experience.current_experience}: {e}")
            print(f"Data in previous: {cl_strategy.evaluator.previous}")
            print(f"Data in initial: {cl_strategy.evaluator.initial}")
        
        print('Training completed')
        
        print('Computing accuracy on the whole test set')
        metrics_performance = cl_strategy.eval(exp_test)
        
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
  
    print(results)
    
    return results

def continual_train(train_stream_online, model, train_epochs = 10, num_domains = 4, lr = 0.0001):
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
    # benchmark = ni_benchmark(
    #     train_dataset=train, 
    #     test_dataset=test,
    #     n_experiences=num_domains, 
    #     task_labels=False,
    #     shuffle=True, 
    #     balance_experiences=True
    # )
    
    # train_stream_online = split_online_stream(
    #     original_stream=benchmark.train_stream,
    #     experience_size=len(train)//16,
    #     drop_last=True
    # )
    
    # test_stream_online = split_online_stream(
    #     original_stream=benchmark.test_stream,
    #     experience_size=len(test)//16,
    #     drop_last=True
    # )

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        #class_accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True),
        forgetting_metrics(experience=True, stream=True),
        bwt_metrics(experience=True, stream=True),
        #AverageForgetting(),
        #forward_transfer_metrics(experience=True, stream=True),
        #cpu_usage_metrics(experience=True),
        AverageBadDecision(),
        OCM(),
        #confusion_matrix_metrics(num_classes=num_class, save_image=False, stream=True),
        #disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        #StreamClassAccuracy(classes=list(range(0,num_class))),
        #StreamAccuracy(),
        # StreamBWT(),
        # StreamForwardTransfer(),
        # ExperienceForwardTransfer(),
        # ExperienceBWT(),
        #bwt_metrics(experience=True, stream=True),
        #forward_transfer_metrics(experience=True, stream=True),
        loggers=[InteractiveLogger(), TextLogger()],
        strict_checks=False
    )
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.001)
    criterion = CrossEntropyLoss()
    train_mb_size = 16
            
    
    cl_strategy = OnlineNaive(
        model, optimizer, criterion,
        train_mb_size=train_mb_size, eval_mb_size=train_mb_size,
        train_passes=train_epochs,
        eval_every=1,
        evaluator=eval_plugin,
        device=device
    )

    print('Starting experiment...')
    print("Training...")
    for experience in train_stream_online:
    #for experience in benchmark.train_stream:
        print("Start of experience ", experience.current_experience)
        print(f"Experience {experience.current_experience} - Number of samples: {len(experience.dataset)}")
        # experiences have an ID that denotes its position in the stream
        eid = experience.current_experience
        print(f"EID={eid}")
        # the experience provides a dataset
        try:
            cl_strategy.train(experience)
        except KeyError as e:
            print(f"KeyError on experience {experience.current_experience}: {e}")
            print(f"Data in previous: {cl_strategy.evaluator.previous}")
            print(f"Data in initial: {cl_strategy.evaluator.initial}")
        
        print('Training completed')
        
        # print('Computing accuracy on the whole test set')
        # metrics_performance = cl_strategy.eval(exp_test)
        
        # keys = list(metrics_performance.keys())
        # line = {}
        # for i in range(len(keys)):
        #     keys_slice = keys[i].split("/")
        #     metric = keys_slice[0]
        #     if metric.find("ClassAcc") != -1:
        #         metric = metric + "_" + keys_slice[-1]
            
        #     phase = keys_slice[1].split("_")[0]
        #     metric = metric + "_" + phase
        #     line["Experiment"] = str(eid)
        #     line[metric] = metrics_performance[keys[i]]
            
        # results_metrics.append(line)
        #results.append(metrics_performance)
        
        # print("Final Results:")
        # print("Dataframe:")
        # domain_data = pd.DataFrame(results_metrics)
        # domain_data.insert(1, "SplitMethod", split_method)
        # domain_data.insert(2, "Client", cli)
        # domain_data.insert(3, "Client", round)
        # print(domain_data)
        
        # if os.path.exists("cl_fl_metrics.csv"):
        #     domain_data.to_csv("cl_fl_metrics.csv", mode="a", header=False, index=False)
        # else:
        #     domain_data.to_csv("cl_fl_metrics.csv", mode="a", header=True, index=False)
            
        # domain_data_values = domain_data.iloc[:,4:]
        # results_dict = {key: domain_data_values[key].mean() for key in domain_data_values.columns.tolist()}
        
        # return results_dict



def continual_test(test_stram_online, model, split_method, cli, round, train_epochs = 10, num_domains = 4, lr = 0.0001):
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
    # benchmark = ni_benchmark(
    #     train_dataset=train, 
    #     test_dataset=test,
    #     n_experiences=num_domains, 
    #     task_labels=False,
    #     shuffle=True, 
    #     balance_experiences=True
    # )
    
    # train_stream_online = split_online_stream(
    #     original_stream=benchmark.train_stream,
    #     experience_size=len(train)//16,
    #     drop_last=True
    # )
    
    # test_stream_online = split_online_stream(
    #     original_stream=benchmark.test_stream,
    #     experience_size=len(test)//16,
    #     drop_last=True
    # )

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        #class_accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True),
        forgetting_metrics(experience=True, stream=True),
        bwt_metrics(experience=True, stream=True),
        #AverageForgetting(),
        #forward_transfer_metrics(experience=True, stream=True),
        #cpu_usage_metrics(experience=True),
        AverageBadDecision(),
        OCM(),
        #confusion_matrix_metrics(num_classes=num_class, save_image=False, stream=True),
        #disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        #StreamClassAccuracy(classes=list(range(0,num_class))),
        #StreamAccuracy(),
        # StreamBWT(),
        # StreamForwardTransfer(),
        # ExperienceForwardTransfer(),
        # ExperienceBWT(),
        #bwt_metrics(experience=True, stream=True),
        #forward_transfer_metrics(experience=True, stream=True),
        loggers=[InteractiveLogger(), TextLogger()],
        strict_checks=False
    )
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.001)
    criterion = CrossEntropyLoss()
    train_mb_size = 1
    
    cl_strategy = OnlineNaive(
        model, optimizer, criterion,
        train_mb_size=train_mb_size, eval_mb_size=train_mb_size,
        train_passes=train_epochs,
        eval_every=1,
        evaluator=eval_plugin,
        device=device
    )
        
    results_metrics = []
    print('Starting experiment...')
    print("Training...")
    for experience in test_stram_online:
    #for experience in benchmark.train_stream:
        print("Start of experience Test ", experience.current_experience)
        print(f"Experience Test {experience.current_experience} - Number of samples: {len(experience.dataset)}")
        # experiences have an ID that denotes its position in the stream
        eid = experience.current_experience
        print(f"EID={eid}")
        # the experience provides a dataset
        
        # print('Computing accuracy on the whole test set')
        metrics_performance = cl_strategy.eval(experience)
        
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
            
        results_metrics.append(line)
        #results.append(metrics_performance)
        
    print("Final Results:")
    print("Dataframe:")
    domain_data = pd.DataFrame(results_metrics)
    domain_data.insert(1, "SplitMethod", split_method)
    domain_data.insert(2, "Client", cli)
    domain_data.insert(3, "Round", round)
    print(domain_data)
        
    if os.path.exists("cl_fl_metrics.csv"):
        domain_data.to_csv("cl_fl_metrics.csv", mode="a", header=False, index=False)
    else:
        domain_data.to_csv("cl_fl_metrics.csv", mode="a", header=True, index=False)
            
    domain_data_values = domain_data.iloc[:,4:]
    results_dict = {key: domain_data_values[key].mean() for key in domain_data_values.columns.tolist()}
        
    return results_dict
    