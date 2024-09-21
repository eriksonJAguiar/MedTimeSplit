from fl_strategy.client_structure import MedicalClientLightning, MedicalClient, MedicalClientContinous
from fl_strategy.server_structure import CustomFedAvg, CustomFedNova, CustomFedProx, CustomScaffold
from utils import utils, partitioning
import pandas as pd
import numpy as np
import flwr
import torch
import os
import json
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument("-mn", "--model_name", required=True)
parser.add_argument("-s", "--strategy", required=True, 
                    help="you could to get the following methods: \
                    FedAvg | FedProx | FedNova | FedScaffold")
parser.add_argument("-sp", "--split", required=True, 
                    help="you could to get the following methods: \
                    SplitMnist | PermutedMnist | MedTimeSplit | None")
args = vars(parser.parse_args())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
client_resources = None
if device.type == "cuda":
    client_resources = {"num_gpus": 1}

root_path = os.path.join("dataset", "MelanomaDB")
csv_path = os.path.join(root_path, "ISIC_2018_dataset.csv")

experiences = 4
num_clients = 5
batch_size = 16
image_size = (224, 224)
model_name = args["model_name"]
split_key = args["split"]
lr = 0.0001
epochs = 10
num_rounds = 50

with open("clients_config/clients_params.json", 'r') as f:
    hyper_params_clients = json.load(f)

num_clients_params = len(hyper_params_clients.keys())
if num_clients > num_clients_params:
    raise ValueError("Enhance the number of parameter in file params.json")

split_method = {
    "SplitMnist": partitioning.load_database_federated_continousSplit(root_path=root_path,
                                                   csv_path=csv_path,
                                                   K=num_clients,
                                                   as_rgb=True,
                                                   image_size=image_size,
                                                ),
    "PermutedMnist": partitioning.load_database_federated_continousPermuted(root_path=root_path,
                                                               csv_path=csv_path,
                                                               K=num_clients,
                                                               as_rgb=True,
                                                               image_size=image_size
                                                            ),
    "MedTimeSplit": partitioning.load_database_federated_continous(root_path=root_path,
                                                           csv_path=csv_path,
                                                           K=num_clients,
                                                           as_rgb=True,
                                                           image_size=image_size,
                                                           hyperparams_client=hyper_params_clients),
}

train, test, num_class = split_method[split_key]

results_fl = []

def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    acc_exp = [num_examples * m["Top1_Acc_Exp_eval"] for num_examples, m in metrics]
    acc_stream = [num_examples * m["Top1_Acc_Stream_eval"] for num_examples, m in metrics]
    forgetting = [num_examples * m["StreamForgetting_eval"] for num_examples, m in metrics]
    btw = [num_examples * m["StreamBWT_eval"] for num_examples, m in metrics]
    abd = [num_examples * m["AverageBadDecision_eval"] for num_examples, m in metrics]
    ocm = [num_examples * m["OCM_eval"] for num_examples, m in metrics]
    
    examples = [num_examples for num_examples, _ in metrics]
    
    results = {
            "val_accuracy_exp": sum(acc_exp)/ sum(examples),
            "val_accuracy_stream": sum(acc_stream)/ sum(examples),
            "val_forgetting": sum(forgetting)/sum(examples),
            "val_btw": sum(btw)/sum(examples),
            "val_adb": sum(abd)/sum(examples),
            "val_ocm": sum(ocm)/sum(examples),
        }

    results_fl.append(results)
    
    return results

def client_fn(cid):
    """read client features

    Returns:
        client_features (MedicalClientContinous): a flower client
    """
    model = utils.make_model_pretrained(model_name=model_name, num_class=num_class)
    
    client_features = MedicalClientContinous(cid=cid,
                                            model=model,
                                            model_name=model_name, 
                                            train_loader=train[int(cid)],
                                            test_loader=test[int(cid)],
                                            split_method=split_key,
                                            num_domain=4, 
                                            lr=lr, 
                                            epoch=epochs,
                                            num_class=num_class,
                                            metrics_file_name="clients_federated_cl.csv"
                                )
    
    return client_features.to_client()

strategy_name = str(args["strategy"])
strategy = None 

if strategy_name == "FedAvg":
    strategy = CustomFedAvg( 
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average
    )
elif strategy_name == "FedProx":
    strategy = CustomFedProx(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        proximal_mu=0.1
    )
elif strategy_name == "FedNova":
    strategy = CustomFedNova(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average
    )
elif strategy_name == "FedScaffold":
    strategy = CustomScaffold(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average
    )
else:
    raise ValueError("strategy key is wrong")

flwr.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=num_clients,
    config=flwr.server.ServerConfig(num_rounds=num_rounds),
    client_resources=client_resources,
    strategy=strategy
)

df_results = pd.DataFrame(results_fl)
df_results.insert(0, "Round", range(len(df_results)))
df_results.insert(1, "Agg", np.repeat(strategy_name, len(df_results)))
df_results.insert(2, "Model",  np.repeat(model_name, len(df_results)))
df_results.insert(3, "SplitMethod",  np.repeat(split_key, len(df_results)))
print(df_results)

if not os.path.exists("federated_learning_cl_results_agg.csv"):
    df_results.to_csv("federated_learning_cl_results_agg.csv", index=False, header=True, mode="a")
else:
    df_results.to_csv("federated_learning_cl_results_agg.csv", index=False, header=False, mode="a")