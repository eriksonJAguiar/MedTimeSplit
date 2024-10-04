
from avalanche.benchmarks import ni_benchmark
from avalanche.logging import TextLogger
import torch
from utils.metrics import OCM, AverageBadDecision

from avalanche.training.supervised.strategy_wrappers_online import OnlineNaive

from avalanche.benchmarks.scenarios.online import split_online_stream

from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics,loss_metrics, timing_metrics, bwt_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.benchmarks.utils import concat_datasets
from avalanche.training.templates import SupervisedTemplate

import pandas as pd
import os
from torch.optim import Adam
from torch.nn import CrossEntropyLoss


device = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Device: {device}")


def run_continual(train, test, model, optimizer, criterion, train_epochs=10, train_mb_size=16, experiences=4, is_train=None):
    """
    Run a continual learning experiment.
    Parameters:
        train (Dataset): The training dataset.
        test (Dataset): The testing dataset.
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (torch.nn.Module): The loss function.
        train_epochs (int, optional): Number of training epochs per experience. Default is 10.
        train_mb_size (int, optional): Mini-batch size for training. Default is 16.
        experiences (int, optional): Number of experiences to split the dataset into. Default is 4.
        is_train (bool, optional): Flag to indicate if the model should be trained. Default is None.
    Returns:
        list: A list of dictionaries containing the performance metrics for each experience.
    """
    if is_train:
        continual_train(train_stream_online, model, train_epochs=train_epochs, num_domains=experiences)
    else:
        return continual_test(test_stream_online, model, split_method="default", cli="default", round=1, train_epochs=train_epochs, num_domains=experiences)
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
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True),
        forgetting_metrics(experience=True, stream=True),
        bwt_metrics(experience=True, stream=True),
        AverageBadDecision(),
        OCM(),
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
        print("Start of experience ", experience.current_experience)
        print(f"Experience {experience.current_experience} - Number of samples: {len(experience.dataset)}")
        eid = experience.current_experience
        print(f"EID={eid}")
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
    
    return results

def continual_train(train_stream_online, model, train_epochs = 10, lr = 0.0001):
    """
    Trains a model using a continual learning strategy on a given training stream.
    Args:
        train_stream_online (Stream): The training data stream for online continual learning.
        model (torch.nn.Module): The model to be trained.
        train_epochs (int, optional): Number of training epochs for each experience. Default is 10.
        lr (float, optional): Learning rate for the optimizer. Default is 0.0001.
    """
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True),
        forgetting_metrics(experience=True, stream=True),
        bwt_metrics(experience=True, stream=True),
        AverageBadDecision(),
        OCM(),
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
        print("Start of experience ", experience.current_experience)
        print(f"Experience {experience.current_experience} - Number of samples: {len(experience.dataset)}")
        eid = experience.current_experience
        print(f"EID={eid}")
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
    """
    Conducts a continual learning test on a given model using an online stream of test experiences.
    Args:
        test_stram_online (iterable): The stream of test experiences.
        model (torch.nn.Module): The model to be evaluated.
        split_method (str): The method used to split the data.
        cli (str): The client identifier.
        round (int): The round number of the experiment.
        train_epochs (int, optional): Number of training epochs per experience. Default is 10.
        num_domains (int, optional): Number of domains in the experiment. Default is 4.
        lr (float, optional): Learning rate for the optimizer. Default is 0.0001.
    Returns:
        dict: A dictionary containing the mean values of the evaluation metrics across all experiences.
    """
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        #class_accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True),
        forgetting_metrics(experience=True, stream=True),
        bwt_metrics(experience=True, stream=True),
        AverageBadDecision(),
        OCM(),
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
    